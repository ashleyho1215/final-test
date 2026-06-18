const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const app = express();
const PORT = 8080;

app.use(express.urlencoded({ extended: true }));
app.use(express.json());
// 告訴伺服器直接到最外層（根目錄）找靜態檔案
app.use(express.static(__dirname));

// 強制將首頁導向根目錄下的 index.html
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.header("Access-Control-Allow-Headers", "Content-Type");
    if (req.method === 'OPTIONS') return res.sendStatus(204);
    next();
});

// ===================================================
// 🧬 1. C4.5 / 資訊增益 動態決策樹訓練引擎
// ===================================================

class ConditionNode {
    constructor(feature, threshold, left, right, infoGain) {
        this.feature = feature;
        this.threshold = threshold;
        this.left = left;    // 小於閾值走左邊
        this.right = right;  // 大於等於閾值走右邊
        this.infoGain = infoGain; // 記錄當時切分的資訊增益
    }
    predict(sleep, steps, mood) {
        const value = this.feature === 'sleep_hours' ? sleep : (this.feature === 'steps' ? steps : mood);
        if (value < this.threshold) return this.left.predict(sleep, steps, mood);
        else return this.right.predict(sleep, steps, mood);
    }
}

class LeafNode {
    constructor(riskLevel) { this.riskLevel = riskLevel; }
    predict() { return this.riskLevel; }
}

// 數學工具：計算資訊熵 (Entropy)
function calculateEntropy(data) {
    if (data.length === 0) return 0;
    const counts = {};
    data.forEach(item => {
        counts[item.risk_label] = (counts[item.risk_label] || 0) + 1;
    });
    let entropy = 0;
    for (const key in counts) {
        const p = counts[key] / data.length;
        entropy -= p * Math.log2(p);
    }
    return entropy;
}

// C4.5 核心：尋找單一特徵的最佳切分閾值與其資訊增益
function findBestThresholdForFeature(data, featureName) {
    let bestGain = -1;
    let bestThreshold = null;
    
    // 找出該特徵所有出現過的值，排序後作為潛在切分點
    const values = data.map(d => d[featureName]).sort((a, b) => a - b);
    const parentEntropy = calculateEntropy(data);

    for (let i = 0; i < values.length - 1; i++) {
        // 取相鄰兩點的中點作為候選閾值
        const midThreshold = (values[i] + values[i + 1]) / 2;
        
        const leftSubset = data.filter(d => d[featureName] < midThreshold);
        const rightSubset = data.filter(d => d[featureName] >= midThreshold);

        if (leftSubset.length === 0 || rightSubset.length === 0) continue;

        // 計算切分後的條件熵
        const leftEntropy = calculateEntropy(leftSubset);
        const rightEntropy = calculateEntropy(rightSubset);
        const conditionalEntropy = (leftSubset.length / data.length) * leftEntropy + (rightSubset.length / data.length) * rightEntropy;

        // 資訊增益 (Information Gain)
        const infoGain = parentEntropy - conditionalEntropy;

        if (infoGain > bestGain) {
            bestGain = infoGain;
            bestThreshold = midThreshold;
        }
    }
    return { infoGain: bestGain, threshold: bestThreshold };
}

// 遞迴建構 C4.5 決策樹 (限制最大深度為 3，避免過度擬合)
function buildC45Tree(data, depth = 1, maxDepth = 3) {
    if (data.length === 0) return new LeafNode('低');

    // 檢查是不是所有資料都屬於同一個標籤 (純度 100%)
    const firstLabel = data[0].risk_label;
    const isPure = data.every(d => d.risk_label === firstLabel);
    if (isPure) return new LeafNode(firstLabel);

    // 達到最大深度時，少數服從多數，直接轉為葉子節點
    if (depth > maxDepth) {
        const counts = {};
        data.forEach(d => counts[d.risk_label] = (counts[d.risk_label] || 0) + 1);
        const majorityLabel = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        return new LeafNode(majorityLabel);
    }

    // 窮舉所有特徵（睡眠、步數、心情），找出資訊增益（Information Gain）最大的那個
    const features = ['sleep_hours', 'steps', 'mood_score'];
    let bestFeature = null;
    let bestThreshold = null;
    let maxGain = -1;

    features.forEach(feat => {
        const result = findBestThresholdForFeature(data, feat);
        if (result.infoGain > maxGain && result.threshold !== null) {
            maxGain = result.infoGain;
            bestFeature = feat;
            bestThreshold = result.threshold;
        }
    });

    // 如果找不到有意義的切分點，直接回傳多數決的葉子
    if (maxGain <= 0 || bestFeature === null) {
        const counts = {};
        data.forEach(d => counts[d.risk_label] = (counts[d.risk_label] || 0) + 1);
        const majorityLabel = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        return new LeafNode(majorityLabel);
    }

    // 列印出後端 ML 引擎計算的資訊增益細節
    console.log(`[C4.5 訓練中] 層次 ${depth}: 最佳特徵為 [${bestFeature}], 最佳門檻值: ${bestThreshold.toFixed(1)}, 獲取資訊增益(IG): ${maxGain.toFixed(4)}`);

    // 根據最佳特徵與門檻將數據切流，向下遞迴生長
    const leftData = data.filter(d => d[bestFeature] < bestThreshold);
    const rightData = data.filter(d => d[bestFeature] >= bestThreshold);

    const leftNode = buildC45Tree(leftData, depth + 1, maxDepth);
    const rightNode = buildC45Tree(rightData, depth + 1, maxDepth);

    return new ConditionNode(bestFeature, bestThreshold, leftNode, rightNode, maxGain);
}

// 全域決策樹根節點
let decisionTreeRoot = null;

// 定義何謂真實的風險標籤（用來當作機器學習的 Y 軸答案基準）
function getTrueLabel(sleep, steps, mood) {
    if (sleep < 5.5 && steps < 3500 && mood <= 4) return '高';
    if (sleep >= 7.0 && steps >= 6000 && mood >= 6) return '低';
    return '中';
}

// 重新從資料庫撈取數據並動態訓練決策樹
function trainDecisionTreeFromDatabase() {
    db.all("SELECT sleep_hours, steps, mood_score FROM health_logs", [], (err, rows) => {
        if (err || !rows || rows.length === 0) {
            console.log("[C4.5 提示] 資料庫內尚無足夠數據，先使用預設基礎樹結構。");
            // 給予一個基礎初始節點避免程式出錯
            decisionTreeRoot = new ConditionNode('sleep_hours', 6.5, new LeafNode('高'), new LeafNode('低'), 0.5);
            return;
        }

        // 為歷史資料標註真實的標籤，作為訓練集 (Dataset)
        const dataset = rows.map(r => ({
            sleep_hours: r.sleep_hours,
            steps: r.steps,
            mood_score: r.mood_score,
            risk_label: getTrueLabel(r.sleep_hours, r.steps, r.mood_score)
        }));

        console.log(`\n===================================================`);
        console.log(`🤖 啟動 C4.5 機器學習演算法 (正在分析資料庫內 ${dataset.length} 筆真實資料)`);
        decisionTreeRoot = buildC45Tree(dataset);
        console.log(`🎉 最佳決策樹模型動態訓練完畢！`);
        console.log(`===================================================\n`);
    });
}

// ===================================================
// 2. SQLite 資料庫初始化
// ===================================================
const db = new sqlite3.Database('health.db');
db.serialize(() => {
    db.run(`
        CREATE TABLE IF NOT EXISTS health_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_date TEXT NOT NULL UNIQUE, 
            sleep_hours REAL NOT NULL,
            steps INTEGER NOT NULL,
            mood_score INTEGER NOT NULL,
            risk_level TEXT
        )
    `, () => {
        // 資料表建立或確認成功後，立刻觸發動態訓練
        trainDecisionTreeFromDatabase();
    });
});

// ===================================================
// 3. RESTful API 端點
// ===================================================

// [GET] /health-logs : 獲取列表並自動以全新訓練出來的 C4.5 樹重新分類
app.get('/health-logs', (req, res) => {
    db.all("SELECT * FROM health_logs ORDER BY log_date DESC", [], (err, rows) => {
        if (err) return res.status(500).json({ error: err.message });
        
        const updatedRows = rows.map(item => {
            // 使用全新的 C4.5 樹動態預測
            const calculatedRisk = decisionTreeRoot.predict(item.sleep_hours, item.steps, item.mood_score);
            if (item.risk_level !== calculatedRisk) {
                item.risk_level = calculatedRisk;
                db.run(`UPDATE health_logs SET risk_level = ? WHERE id = ?`, [calculatedRisk, item.id]);
            }
            return item;
        });
        res.json(updatedRows);
    });
});

// [POST] 新增資料
app.post('/health-logs', (req, res) => {
    const { log_date, sleep_hours, steps, mood_score } = req.body;
    const sql = `INSERT INTO health_logs (log_date, sleep_hours, steps, mood_score, risk_level) VALUES (?, ?, ?, ?, NULL)`;
    db.run(sql, [log_date, sleep_hours, steps, mood_score], function(err) {
        if (err) {
            if (err.message.includes("UNIQUE constraint failed")) {
                return res.status(409).json({ message: `您在 ${log_date} 已經填寫過日誌囉！` });
            }
            return res.status(500).json({ error: err.message });
        }
        // ✨ 每當使用者新增或刪除資料，資料庫數據改變了，就重新訓練一次樹，保持最優解！
        trainDecisionTreeFromDatabase();
        res.status(201).json({ status: "success", message: "健康日誌新增成功！" });
    });
});

// [GET] /health-logs/risk : 依據近7天趨勢進行 C4.5 風險判定
app.get('/health-logs/risk', (req, res) => {
    const trendSql = `
        SELECT AVG(sleep_hours) as avg_sleep, AVG(steps) as avg_steps, MIN(mood_score) as min_mood, COUNT(id) as total_days
        FROM (SELECT * FROM health_logs ORDER BY log_date DESC LIMIT 7)
    `;
    db.get(trendSql, [], (err, row) => {
        if (err) return res.status(500).json({ error: err.message });
        if (!row || row.total_days === 0) return res.status(400).json({ error: "無資料" });

        const avgSleep = row.avg_sleep;
        const avgSteps = row.avg_steps;
        const minMood = row.min_mood;

        // 呼叫 C4.5 訓練出來的根節點進行預測
        const calculatedRisk = decisionTreeRoot.predict(avgSleep, avgSteps, minMood);

        const updateSql = `UPDATE health_logs SET risk_level = ? WHERE log_date = (SELECT MAX(log_date) FROM health_logs)`;
        db.run(updateSql, [calculatedRisk], () => {
            res.json({
                status: "success",
                risk_level: calculatedRisk,
                analysis: { days_counted: row.total_days, avg_sleep: avgSleep.toFixed(1), avg_steps: Math.round(avgSteps), min_mood: minMood }
            });
        });
    });
});

// 刪除端點
app.delete('/health-logs/:id', (req, res) => {
    db.run(`DELETE FROM health_logs WHERE id = ?`, [req.params.id], function(err) {
        if (err) return res.status(500).json({ error: err.message });
        trainDecisionTreeFromDatabase(); // 刪除資料也重新訓練
        res.json({ message: "刪除成功" });
    });
});

// 修改端點 (PUT)
app.put('/health-logs/:id', (req, res) => {
    const { log_date, sleep_hours, steps, mood_score } = req.body;
    db.run(
        `UPDATE health_logs SET log_date=?, sleep_hours=?, steps=?, mood_score=?, risk_level=NULL WHERE id=?`,
        [log_date, sleep_hours, steps, mood_score, req.params.id],
        function(err) {
            if (err) return res.status(500).json({ error: err.message });
            trainDecisionTreeFromDatabase(); // 修改資料也重新訓練
            res.json({ status: "success", message: "日誌資料修改成功！" });
        }
    );
});

app.listen(8080, () => console.log(`C4.js 伺服器正執行於：http://localhost:8080`));