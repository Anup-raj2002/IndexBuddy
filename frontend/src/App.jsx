import React, { useEffect, useState } from "react";

const STEP_META = [
  { num: 1, label: "Schema & Sample" },
  { num: 2, label: "Run No-Index Baseline" },
  { num: 3, label: "ML Recommendation" },
  { num: 4, label: "Confirm CREATE INDEX" },
  { num: 5, label: "Run With Index" },
  { num: 6, label: "Predicted vs Actual" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("wizard"); // 'wizard' | 'evaluation'
  const [step, setStep] = useState(1);
  const [dbStatus, setDbStatus] = useState(null);
  const [schemaData, setSchemaData] = useState(null);
  const [selectedDemoId, setSelectedDemoId] = useState("demo_t11_composite");
  const [sqlText, setSqlText] = useState("");
  const [threshold, setThreshold] = useState(1.5);

  const [step2Result, setStep2Result] = useState(null);
  const [step3Result, setStep3Result] = useState(null);
  const [step4Result, setStep4Result] = useState(null);
  const [step5Result, setStep5Result] = useState(null);
  const [step6Result, setStep6Result] = useState(null);
  const [evalSummary, setEvalSummary] = useState(null);

  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [toastMsg, setToastMsg] = useState("");

  async function fetchStatus() {
    try {
      const r = await fetch("/api/status");
      const data = await r.json();
      setDbStatus(data);
    } catch (e) {
      setDbStatus({ connected: false, error: String(e) });
    }
  }

  async function fetchStep1() {
    setLoading(true);
    setErrorMsg("");
    try {
      const r = await fetch("/api/step1/schema");
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Failed to load schema");
      setSchemaData(data);
      if (!sqlText && data.demo_queries?.length) {
        setSqlText(data.demo_queries[0].sql);
      }
    } catch (e) {
      setErrorMsg(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  async function fetchEvaluation() {
    setLoading(true);
    setErrorMsg("");
    try {
      const r = await fetch("/api/evaluation");
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Failed to load evaluation");
      setEvalSummary(data);
    } catch (e) {
      setErrorMsg(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchStatus();
    fetchStep1();
  }, []);

  useEffect(() => {
    if (activeTab === "evaluation" && !evalSummary) {
      fetchEvaluation();
    }
  }, [activeTab]);

  function handleSelectDemoQuery(demoId) {
    setSelectedDemoId(demoId);
    const found = schemaData?.demo_queries?.find((d) => d.id === demoId);
    if (found) {
      setSqlText(found.sql);
      setStep2Result(null);
      setStep3Result(null);
      setStep4Result(null);
      setStep5Result(null);
      setStep6Result(null);
    }
  }

  async function handleResetDemo() {
    setLoading(true);
    setErrorMsg("");
    try {
      const r = await fetch("/api/reset-demo", { method: "POST" });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Reset failed");
      setStep2Result(null);
      setStep3Result(null);
      setStep4Result(null);
      setStep5Result(null);
      setStep6Result(null);
      setStep(1);
      setToastMsg(
        `Demo reset complete. Dropped secondary indexes: ${
          data.dropped_indexes?.length ? data.dropped_indexes.join(", ") : "none"
        }`
      );
      await fetchStatus();
    } catch (e) {
      setErrorMsg(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  async function handleRunBaseline() {
    setLoading(true);
    setErrorMsg("");
    setToastMsg("");
    try {
      const r = await fetch("/api/step2/run-baseline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sql: sqlText,
          runs: 10,
          warmup: 1,
          drop_secondary_indexes_first: true,
        }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Baseline execution failed");
      setStep2Result(data);
      setStep3Result(null);
      setStep4Result(null);
      setStep5Result(null);
      setStep6Result(null);
      await fetchStatus();
    } catch (e) {
      setErrorMsg(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  async function handleGetRecommendation() {
    setLoading(true);
    setErrorMsg("");
    try {
      const r = await fetch("/api/step3/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sql: sqlText,
          threshold: Number(threshold),
        }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Recommendation failed");
      setStep3Result(data);
    } catch (e) {
      setErrorMsg(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  async function handleConfirmCreateIndex(candidateObj) {
    if (!candidateObj) return;
    setLoading(true);
    setErrorMsg("");
    try {
      const r = await fetch("/api/step4/create-index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          table: candidateObj.table,
          index_name: candidateObj.index_name,
          ddl: candidateObj.ddl,
        }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "CREATE INDEX failed");
      setStep4Result({ ...data, candidate: candidateObj });
      await fetchStatus();
    } catch (e) {
      setErrorMsg(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  async function handleRunAfterIndex() {
    setLoading(true);
    setErrorMsg("");
    try {
      const r = await fetch("/api/step5/run-after", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sql: sqlText,
          runs: 10,
          warmup: 1,
          drop_secondary_indexes_first: false,
        }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Post-index execution failed");
      setStep5Result(data);
      await fetchStatus();
    } catch (e) {
      setErrorMsg(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  async function handleBuildComparison() {
    if (!step2Result || !step3Result || !step5Result) return;
    setLoading(true);
    setErrorMsg("");
    try {
      const cand =
        step4Result?.candidate ||
        step3Result.recommended_candidate ||
        step3Result.top_fallback_candidate;
      const r = await fetch("/api/step6/compare-and-save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sql: sqlText,
          chosen_index_name: cand?.index_name || null,
          chosen_columns: cand?.columns_display || null,
          ddl_executed: cand?.ddl || null,
          predicted_speedup: cand?.predicted_speedup || 1.0,
          estimated_selectivity: cand?.estimated_selectivity || 1.0,
          threshold: Number(threshold),
          before_measurement: step2Result.measurement,
          after_measurement: step5Result.measurement,
          ranked_candidates: step3Result.ranked_candidates || [],
        }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Comparison failed");
      setStep6Result(data);
      setStep(6);
    } catch (e) {
      setErrorMsg(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  const currentDemoMeta = schemaData?.demo_queries?.find(
    (d) => d.id === selectedDemoId
  );

  return (
    <div className="app-shell">
      {/* Persistent Top Database Status Bar */}
      <header className="top-status-bar">
        <div className="brand-group">
          <div className="brand-logo">IB</div>
          <div>
            <div className="brand-title">IndexBuddy ML</div>
            <div className="brand-subtitle">
              XGBoost Speedup Regressor &bull; Live MySQL 8.x Automatic Index Advisor
            </div>
          </div>
        </div>

        <div className="status-pills">
          <div className="status-pill">
            <span
              className={`pulse-dot ${dbStatus?.connected ? "" : "offline"}`}
            />
            <span>
              {dbStatus?.connected
                ? `MySQL ${dbStatus.mysql_version} Connected (${dbStatus.host}:${dbStatus.port})`
                : "MySQL Disconnected"}
            </span>
          </div>
          <div className="status-pill">
            <span>Table:</span>
            <strong className="mono">
              {dbStatus?.table_name || "orders_benchmark"}
            </strong>
          </div>
          <div className="status-pill">
            <span>Total Rows:</span>
            <strong className="mono">
              {dbStatus?.total_rows
                ? Number(dbStatus.total_rows).toLocaleString()
                : "0"}
            </strong>
          </div>
          <div className="status-pill">
            <span>Secondary Indexes:</span>
            <strong className="mono">
              {dbStatus?.secondary_indexes?.length
                ? dbStatus.secondary_indexes.join(", ")
                : "0 (None)"}
            </strong>
          </div>
        </div>

        <div className="top-actions">
          <div className="mode-tabs">
            <button
              className={`mode-tab ${activeTab === "wizard" ? "active" : ""}`}
              onClick={() => setActiveTab("wizard")}
            >
              6-Step Live Demo
            </button>
            <button
              className={`mode-tab ${activeTab === "evaluation" ? "active" : ""}`}
              onClick={() => setActiveTab("evaluation")}
            >
              SHAP &amp; Evaluation Lab
            </button>
          </div>
          <button
            className="btn btn-danger-outline"
            onClick={handleResetDemo}
            disabled={loading}
          >
            Reset Demo
          </button>
        </div>
      </header>

      {errorMsg && (
        <div className="alert alert-error">
          <strong>Error:</strong> {errorMsg}
        </div>
      )}
      {toastMsg && <div className="alert alert-success">{toastMsg}</div>}

      {activeTab === "evaluation" ? (
        <EvaluationLab
          evalSummary={evalSummary}
          loading={loading}
          onRefresh={fetchEvaluation}
        />
      ) : (
        <>
          {/* 6-Step Wizard Navigation */}
          <nav className="stepper-bar">
            {STEP_META.map((s) => {
              const isCompleted =
                (s.num === 1 && schemaData) ||
                (s.num === 2 && step2Result) ||
                (s.num === 3 && step3Result) ||
                (s.num === 4 && step4Result) ||
                (s.num === 5 && step5Result) ||
                (s.num === 6 && step6Result);
              return (
                <button
                  key={s.num}
                  className={`step-item ${step === s.num ? "active" : ""} ${
                    isCompleted ? "completed" : ""
                  }`}
                  onClick={() => setStep(s.num)}
                >
                  <div className="step-num">
                    <span>Step {s.num}</span>
                    {isCompleted && <span>&#10003;</span>}
                  </div>
                  <div className="step-label">{s.label}</div>
                </button>
              );
            })}
          </nav>

          {/* STEP 1: Schema & Sample Records */}
          {step === 1 && (
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2 className="panel-title">
                    Step 1: Live `orders_benchmark` Schema, Cardinality &amp; Skew
                  </h2>
                  <p className="panel-subtitle">
                    Connected live to your local MySQL database{" "}
                    <strong>`indexbuddy`</strong> &rarr; table{" "}
                    <strong>`orders_benchmark` (1,000,000 rows)</strong>.
                  </p>
                </div>
                <button
                  className="btn btn-primary"
                  onClick={() => setStep(2)}
                >
                  Next: Step 2 (Run a Slow Query) &rarr;
                </button>
              </div>

              <div
                className="alert"
                style={{
                  background: "rgba(59, 130, 246, 0.12)",
                  border: "1px solid rgba(59, 130, 246, 0.4)",
                  color: "#dbeafe",
                }}
              >
                <strong>What are you looking at in Step 1?</strong>
                <div style={{ marginTop: 4 }}>
                  We created a database called <code>indexbuddy</code> in your
                  local MySQL server with a table named{" "}
                  <code>orders_benchmark</code> containing{" "}
                  <strong>1,000,000 e-commerce orders</strong> and{" "}
                  <strong>NO indexes</strong> (except the primary key{" "}
                  <code>order_id</code>).
                  <br />
                  &bull; <strong>Cardinality</strong> (below left) means{" "}
                  <em>"how many unique values exist in this column"</em>:{" "}
                  <code>customer_id</code> has <strong>99,994</strong> unique
                  customers (great for indexing!), whereas <code>status</code>{" "}
                  only has <strong>5</strong> unique values (bad for indexing if
                  filtering on <code>'completed'</code>!).
                </div>
              </div>

              {schemaData && (
                <>
                  <div className="kpi-grid">
                    <div className="kpi-card">
                      <div className="kpi-label">Benchmark Table</div>
                      <div className="kpi-value">{schemaData.table_name}</div>
                      <div className="kpi-sub">InnoDB Clustered PK (order_id)</div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-label">Live Row Count</div>
                      <div className="kpi-value">
                        {Number(schemaData.total_rows).toLocaleString()}
                      </div>
                      <div className="kpi-sub">
                        Avg Row Length: {schemaData.avg_row_length} B
                      </div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-label">Table Data Size</div>
                      <div className="kpi-value">{schemaData.data_length_mb} MB</div>
                      <div className="kpi-sub">100% fits in InnoDB Buffer Pool</div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-label">Held-Out Demo Queries</div>
                      <div className="kpi-value">
                        {schemaData.demo_queries?.length || 5}
                      </div>
                      <div className="kpi-sub">Templates held out from training</div>
                    </div>
                  </div>

                  <div className="grid-2" style={{ marginBottom: 18 }}>
                    <div>
                      <h3 style={{ fontSize: 15, marginBottom: 10 }}>
                        Column Schema &amp; Live Cardinalities (`COUNT DISTINCT`)
                      </h3>
                      <div className="table-wrap">
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Column</th>
                              <th>Type</th>
                              <th>Key</th>
                              <th>Cardinality</th>
                              <th>Distinct %</th>
                              <th>Null %</th>
                              <th>Width (B)</th>
                            </tr>
                          </thead>
                          <tbody>
                            {schemaData.schema.map((col) => (
                              <tr key={col.column_name}>
                                <td className="mono">
                                  <strong>{col.column_name}</strong>
                                </td>
                                <td className="mono">{col.column_type}</td>
                                <td>
                                  {col.column_key ? (
                                    <span className="badge badge-blue">
                                      {col.column_key}
                                    </span>
                                  ) : (
                                    "—"
                                  )}
                                </td>
                                <td className="mono">
                                  {Number(col.cardinality).toLocaleString()}
                                </td>
                                <td className="mono">{col.distinct_ratio_pct}%</td>
                                <td className="mono">{col.null_fraction_pct}%</td>
                                <td className="mono">{col.byte_width}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    <div>
                      <h3 style={{ fontSize: 15, marginBottom: 10 }}>
                        Live `status` Skew Distribution (Why Low-Selectivity Fails)
                      </h3>
                      <div
                        className="panel"
                        style={{ padding: 16, marginBottom: 0 }}
                      >
                        {schemaData.status_distribution.map((s) => (
                          <div className="bar-row" key={s.status}>
                            <div className="bar-meta">
                              <span className="mono">
                                <strong>'{s.status}'</strong>
                              </span>
                              <span className="mono">
                                {Number(s.row_count).toLocaleString()} rows (
                                {s.pct}%)
                              </span>
                            </div>
                            <div className="bar-track">
                              <div
                                className="bar-fill"
                                style={{
                                  width: `${Math.min(100, Number(s.pct))}%`,
                                  background:
                                    Number(s.pct) > 50
                                      ? "var(--danger)"
                                      : "var(--accent)",
                                }}
                              />
                            </div>
                          </div>
                        ))}
                        <p
                          style={{
                            fontSize: 12,
                            color: "var(--text-muted)",
                            marginTop: 10,
                          }}
                        >
                          Notice that <code>status = 'completed'</code> matches{" "}
                          <strong>~70% of the table</strong>. An index on{" "}
                          <code>status</code> for <code>'completed'</code> is
                          unselective, so the ML model should recommend{" "}
                          <strong>No Index</strong>.
                        </p>
                      </div>
                    </div>
                  </div>

                  <h3 style={{ fontSize: 15, marginBottom: 10 }}>
                    Live Sample Rows (`SELECT * FROM orders_benchmark LIMIT 12`)
                  </h3>
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          {schemaData.sample_records[0] &&
                            Object.keys(schemaData.sample_records[0]).map((k) => (
                              <th key={k}>{k}</th>
                            ))}
                        </tr>
                      </thead>
                      <tbody>
                        {schemaData.sample_records.map((row, idx) => (
                          <tr key={idx}>
                            {Object.values(row).map((v, j) => (
                              <td key={j} className="mono">
                                {v === null ? "NULL" : String(v)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          )}

          {/* STEP 2: Baseline Execution (No Index) */}
          {step === 2 && (
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2 className="panel-title">
                    Step 2: Choose Held-Out Demo Query &amp; Run With NO Index
                  </h2>
                  <p className="panel-subtitle">
                    All preset demo queries belong to templates (`T11`, `T03`,
                    `T07`, `T16`, `T19`) strictly held out from XGBoost training.
                  </p>
                </div>
                <button
                  className="btn btn-primary"
                  onClick={handleRunBaseline}
                  disabled={loading || !sqlText.trim()}
                >
                  {loading
                    ? "Running 10 Executions on MySQL..."
                    : "Run Query With NO Index (10 Runs, 1 Warm-Up)"}
                </button>
              </div>

              <div
                className="alert"
                style={{
                  background: "rgba(59, 130, 246, 0.12)",
                  border: "1px solid rgba(59, 130, 246, 0.4)",
                  color: "#dbeafe",
                }}
              >
                <strong>In Simple Words — What happens in Step 2?</strong>
                <div style={{ marginTop: 4 }}>
                  Right now, our table of <strong>1,000,000 orders</strong> has{" "}
                  <strong>no index</strong>. When you click{" "}
                  <strong>"Run Query With NO Index"</strong>, MySQL has to check
                  every single row from row #1 to row #1,000,000 (like flipping
                  through every page of a 1,000-page book because there is no
                  index at the back). You will see it takes almost{" "}
                  <strong>1 full second (~900 ms)</strong>.
                </div>
              </div>

              <label
                style={{
                  display: "block",
                  fontSize: 12.5,
                  color: "var(--text-muted)",
                  marginBottom: 6,
                }}
              >
                Preset Held-Out Demo Queries:
              </label>
              <select
                className="sql-select"
                value={selectedDemoId}
                onChange={(e) => handleSelectDemoQuery(e.target.value)}
              >
                {schemaData?.demo_queries?.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.label}
                  </option>
                ))}
              </select>

              {currentDemoMeta && (
                <div
                  style={{
                    fontSize: 12.5,
                    color: "#93c5fd",
                    marginBottom: 10,
                  }}
                >
                  <span className="badge badge-blue" style={{ marginRight: 8 }}>
                    Held-Out {currentDemoMeta.template_id}
                  </span>
                  {currentDemoMeta.description}
                </div>
              )}

              <label
                style={{
                  display: "block",
                  fontSize: 12.5,
                  color: "var(--text-muted)",
                  marginBottom: 6,
                }}
              >
                Editable SQL Query:
              </label>
              <textarea
                className="sql-editor"
                value={sqlText}
                onChange={(e) => setSqlText(e.target.value)}
              />

              {step2Result && (
                <>
                  <div className="kpi-grid" style={{ marginTop: 14 }}>
                    <div className="kpi-card">
                      <div className="kpi-label">
                        Median Execution Time (No Index)
                      </div>
                      <div className="kpi-value" style={{ color: "#f87171" }}>
                        {step2Result.measurement.median_time_ms.toFixed(2)} ms
                      </div>
                      <div className="kpi-sub">
                        Min: {step2Result.measurement.min_time_ms.toFixed(1)} ms
                        &bull; Max:{" "}
                        {step2Result.measurement.max_time_ms.toFixed(1)} ms
                      </div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-label">EXPLAIN Rows Examined</div>
                      <div className="kpi-value">
                        {Number(
                          step2Result.measurement.rows_examined
                        ).toLocaleString()}
                      </div>
                      <div className="kpi-sub">Full clustered table scan</div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-label">MySQL Access Type</div>
                      <div className="kpi-value">
                        <span className="badge badge-red">
                          {step2Result.measurement.access_type}
                        </span>
                      </div>
                      <div className="kpi-sub">
                        Extra: {step2Result.measurement.extra || "Using where"}
                      </div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-label">Secondary Key Used</div>
                      <div className="kpi-value">
                        {step2Result.measurement.key_used || "NONE"}
                      </div>
                      <div className="kpi-sub">
                        Active indexes:{" "}
                        {step2Result.active_secondary_indexes.length}
                      </div>
                    </div>
                  </div>

                  {/* Parsed AST Summary */}
                  <div
                    className="panel"
                    style={{
                      background: "var(--bg-card-alt)",
                      padding: 14,
                      marginBottom: 16,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 13,
                        fontWeight: 700,
                        marginBottom: 8,
                      }}
                    >
                      `sqlglot` MySQL AST Extraction:
                    </div>
                    <div
                      style={{
                        display: "flex",
                        flexWrap: "wrap",
                        gap: 10,
                        fontSize: 12.5,
                      }}
                    >
                      <span className="badge badge-blue">
                        Table: {step2Result.parsed.primary_table}
                      </span>
                      <span className="badge badge-green">
                        WHERE:{" "}
                        {step2Result.parsed.where_columns.join(", ") || "none"}
                      </span>
                      <span className="badge badge-amber">
                        Predicates:{" "}
                        {JSON.stringify(step2Result.parsed.predicate_types)}
                      </span>
                      {step2Result.parsed.join_columns.length > 0 && (
                        <span className="badge badge-blue">
                          JOIN: {step2Result.parsed.join_columns.join(", ")}
                        </span>
                      )}
                      {step2Result.parsed.group_by_columns.length > 0 && (
                        <span className="badge badge-blue">
                          GROUP BY:{" "}
                          {step2Result.parsed.group_by_columns.join(", ")}
                        </span>
                      )}
                      {step2Result.parsed.order_by_columns.length > 0 && (
                        <span className="badge badge-blue">
                          ORDER BY:{" "}
                          {step2Result.parsed.order_by_columns.join(", ")}
                        </span>
                      )}
                    </div>
                    <div
                      className="mono"
                      style={{
                        fontSize: 11.5,
                        color: "var(--text-muted)",
                        marginTop: 8,
                      }}
                    >
                      Canonical Template: {step2Result.parsed.template_sql}
                    </div>
                  </div>

                  <div className="grid-2-equal">
                    <div>
                      <h3 style={{ fontSize: 14.5, marginBottom: 8 }}>
                        Live `EXPLAIN` Plan (No Index)
                      </h3>
                      <ExplainTable rows={step2Result.measurement.explain_rows} />
                    </div>
                    <div>
                      <h3 style={{ fontSize: 14.5, marginBottom: 8 }}>
                        Live `EXPLAIN ANALYZE` Execution Tree (No Index)
                      </h3>
                      <pre className="code-box">
                        {step2Result.measurement.explain_analyze}
                      </pre>
                    </div>
                  </div>
                </>
              )}

              <div className="step-footer">
                <button
                  className="btn btn-secondary"
                  onClick={() => setStep(1)}
                >
                  &larr; Back to Step 1
                </button>
                <button
                  className="btn btn-primary"
                  onClick={() => setStep(3)}
                  disabled={!step2Result}
                >
                  Next: Step 3 (Get ML Recommendation) &rarr;
                </button>
              </div>
            </div>
          )}

          {/* STEP 3: Get ML Recommendation */}
          {step === 3 && (
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2 className="panel-title">
                    Step 3: XGBoost Candidate Ranking &amp; Speedup Prediction
                  </h2>
                  <p className="panel-subtitle">
                    Scores all generated 1-column and 2-column B-tree candidates
                    exclusively with the trained XGBoost regressor. Recommends an
                    index iff predicted speedup &ge; threshold.
                  </p>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                  <div style={{ fontSize: 12.5 }}>
                    <label style={{ color: "var(--text-muted)", marginRight: 8 }}>
                      Speedup Threshold: <strong>{threshold}x</strong>
                    </label>
                    <input
                      type="range"
                      min="1.1"
                      max="5.0"
                      step="0.1"
                      value={threshold}
                      onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    />
                  </div>
                  <button
                    className="btn btn-primary"
                    onClick={handleGetRecommendation}
                    disabled={loading}
                  >
                    {loading ? "Scoring Candidates..." : "Get Recommendation"}
                  </button>
                </div>
              </div>

              <pre className="code-box" style={{ marginBottom: 16 }}>
                {sqlText}
              </pre>

              <div
                className="alert"
                style={{
                  background: "rgba(59, 130, 246, 0.12)",
                  border: "1px solid rgba(59, 130, 246, 0.4)",
                  color: "#dbeafe",
                }}
              >
                <strong>In Simple Words — What happens in Step 3?</strong>
                <div style={{ marginTop: 4 }}>
                  Instead of guessing which column to index, our{" "}
                  <strong>Machine Learning model (XGBoost)</strong> looks at all
                  possible indexes for your query and predicts{" "}
                  <strong>how many times faster</strong> each one will be. If the
                  best index makes the query at least <strong>1.5x faster</strong>
                  , it recommends creating it; otherwise it says{" "}
                  <strong>"No Index Needed"</strong>.
                </div>
              </div>

              {step3Result && (
                <>
                  <div
                    className={`alert ${
                      step3Result.recommend_create_index
                        ? "alert-success"
                        : "alert-warning"
                    }`}
                  >
                    <strong>
                      {step3Result.recommend_create_index
                        ? "ML Recommendation: CREATE INDEX"
                        : "ML Recommendation: NO INDEX (Below Speedup Threshold)"}
                    </strong>
                    <div style={{ marginTop: 4 }}>{step3Result.reason}</div>
                  </div>

                  <h3 style={{ fontSize: 15, marginBottom: 10 }}>
                    Candidate Indexes Ranked by XGBoost Predicted Speedup
                  </h3>
                  <div className="table-wrap" style={{ marginBottom: 18 }}>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Rank</th>
                          <th>Candidate Columns</th>
                          <th>Type</th>
                          <th>Source Clauses</th>
                          <th>XGBoost Predicted Speedup</th>
                          <th>Predicted log(Speedup)</th>
                          <th>Est. Effective Selectivity</th>
                          <th>Est. Index Size</th>
                          <th>Decision</th>
                        </tr>
                      </thead>
                      <tbody>
                        {step3Result.ranked_candidates.map((c) => (
                          <tr
                            key={c.index_name}
                            className={
                              step3Result.recommended_candidate?.index_name ===
                              c.index_name
                                ? "highlight-row"
                                : ""
                            }
                          >
                            <td className="mono">#{c.rank}</td>
                            <td className="mono">
                              <strong>({c.columns_display})</strong>
                            </td>
                            <td>
                              <span className="badge badge-blue">
                                {c.is_composite ? "2-Col Composite" : "1-Col Single"}
                              </span>
                            </td>
                            <td className="mono">
                              {c.source_clauses.join(", ")}
                            </td>
                            <td className="mono">
                              <strong
                                style={{
                                  color: c.meets_threshold
                                    ? "#34d399"
                                    : "#f87171",
                                  fontSize: 14,
                                }}
                              >
                                {c.predicted_speedup.toFixed(2)}x
                              </strong>
                            </td>
                            <td className="mono">
                              {c.predicted_log_speedup.toFixed(3)}
                            </td>
                            <td className="mono">
                              {c.estimated_selectivity_pct}%
                            </td>
                            <td className="mono">
                              {c.estimated_index_size_mb} MB
                            </td>
                            <td>
                              {step3Result.recommended_candidate?.index_name ===
                              c.index_name ? (
                                <span className="badge badge-green">
                                  CHOSEN TOP-1
                                </span>
                              ) : c.meets_threshold ? (
                                <span className="badge badge-blue">
                                  Viable (&ge;{step3Result.threshold}x)
                                </span>
                              ) : (
                                <span className="badge badge-red">
                                  Rejected (&lt;{step3Result.threshold}x)
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Top candidate SHAP attribution */}
                  {step3Result.ranked_candidates[0]?.shap_top_factors?.length >
                    0 && (
                    <div
                      className="panel"
                      style={{
                        background: "var(--bg-card-alt)",
                        padding: 16,
                        marginBottom: 0,
                      }}
                    >
                      <h4 style={{ fontSize: 14, marginBottom: 10 }}>
                        SHAP Feature Attributions for Top Candidate `(
                        {step3Result.ranked_candidates[0].columns_display})`
                      </h4>
                      <div className="grid-2-equal">
                        {step3Result.ranked_candidates[0].shap_top_factors.map(
                          (f) => (
                            <div
                              key={f.feature}
                              style={{
                                display: "flex",
                                justifyContent: "space-between",
                                padding: "6px 10px",
                                background: "var(--bg-code)",
                                borderRadius: 8,
                                fontSize: 12.5,
                              }}
                            >
                              <span className="mono">
                                {f.feature} ={" "}
                                <strong>{Number(f.feature_value).toFixed(4)}</strong>
                              </span>
                              <span
                                className="mono"
                                style={{
                                  color:
                                    f.shap_log_impact >= 0
                                      ? "#34d399"
                                      : "#f87171",
                                }}
                              >
                                {f.shap_log_impact >= 0 ? "+" : ""}
                                {f.shap_log_impact.toFixed(3)} log-spd
                              </span>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}

              <div className="step-footer">
                <button
                  className="btn btn-secondary"
                  onClick={() => setStep(2)}
                >
                  &larr; Back to Step 2
                </button>
                <button
                  className="btn btn-primary"
                  onClick={() => setStep(4)}
                  disabled={!step3Result}
                >
                  Next: Step 4 (Review &amp; Confirm DDL) &rarr;
                </button>
              </div>
            </div>
          )}

          {/* STEP 4: Confirm CREATE INDEX DDL */}
          {step === 4 && (
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2 className="panel-title">
                    Step 4: Review Generated DDL &amp; Confirm Index Creation
                  </h2>
                  <p className="panel-subtitle">
                    IndexBuddy never modifies your MySQL schema without explicit
                    confirmation. Click Confirm below to execute `CREATE INDEX` on
                    live MySQL.
                  </p>
                </div>
              </div>

              <div
                className="alert"
                style={{
                  background: "rgba(59, 130, 246, 0.12)",
                  border: "1px solid rgba(59, 130, 246, 0.4)",
                  color: "#dbeafe",
                }}
              >
                <strong>In Simple Words — What happens in Step 4?</strong>
                <div style={{ marginTop: 4 }}>
                  The AI wrote the SQL command (<code>CREATE INDEX ...</code>) to
                  add the winning index to your MySQL table. For safety, it waits
                  for your permission—when you click{" "}
                  <strong>"Confirm &amp; Execute CREATE INDEX"</strong>, MySQL
                  actually builds the index on the 1,000,000 rows (~4 seconds).
                </div>
              </div>

              {!step3Result ? (
                <div className="alert alert-warning">
                  Please complete Step 3 first to generate a candidate
                  recommendation.
                </div>
              ) : step3Result.recommend_create_index &&
                step3Result.recommended_candidate ? (
                <>
                  <div className="alert alert-success">
                    <strong>Chosen Candidate:</strong> `
                    {step3Result.recommended_candidate.index_name}` on `(
                    {step3Result.recommended_candidate.columns_display})` &bull;
                    Predicted Speedup:{" "}
                    <strong>
                      {step3Result.recommended_candidate.predicted_speedup.toFixed(
                        2
                      )}
                      x
                    </strong>
                  </div>

                  <h3 style={{ fontSize: 14.5, marginBottom: 8 }}>
                    Generated MySQL `CREATE INDEX` DDL:
                  </h3>
                  <pre className="code-box" style={{ marginBottom: 16 }}>
                    {step3Result.recommended_candidate.ddl}
                  </pre>

                  <button
                    className="btn btn-success"
                    onClick={() =>
                      handleConfirmCreateIndex(
                        step3Result.recommended_candidate
                      )
                    }
                    disabled={loading}
                  >
                    {loading
                      ? "Creating B-Tree Index on 1,000,000 Rows..."
                      : "Confirm & Execute CREATE INDEX on Live MySQL"}
                  </button>
                </>
              ) : (
                <>
                  <div className="alert alert-warning">
                    <strong>No Index Recommended by XGBoost Model</strong>
                    <div style={{ marginTop: 4 }}>{step3Result.reason}</div>
                  </div>
                  {step3Result.top_fallback_candidate && (
                    <div
                      className="panel"
                      style={{ background: "var(--bg-card-alt)", padding: 16 }}
                    >
                      <h4 style={{ fontSize: 14, marginBottom: 8 }}>
                        Professor Verification Option (Test Creating the
                        Rejected Index Anyway):
                      </h4>
                      <p
                        style={{
                          fontSize: 12.5,
                          color: "var(--text-muted)",
                          marginBottom: 10,
                        }}
                      >
                        Want to prove empirically why the ML model rejected this
                        candidate? Click below to force-create `
                        {step3Result.top_fallback_candidate.index_name}` and
                        measure the real post-index query speed in Step 5 &amp; 6.
                      </p>
                      <pre className="code-box" style={{ marginBottom: 12 }}>
                        {step3Result.top_fallback_candidate.ddl}
                      </pre>
                      <button
                        className="btn btn-secondary"
                        onClick={() =>
                          handleConfirmCreateIndex(
                            step3Result.top_fallback_candidate
                          )
                        }
                        disabled={loading}
                      >
                        {loading
                          ? "Creating Index on MySQL..."
                          : "Force-Create Rejected Index to Measure Empirical Result"}
                      </button>
                    </div>
                  )}
                </>
              )}

              {step4Result && (
                <div className="alert alert-success" style={{ marginTop: 16 }}>
                  <strong>Index Created Successfully on Live MySQL!</strong>
                  <div className="mono" style={{ marginTop: 4, fontSize: 12.5 }}>
                    Executed: {step4Result.ddl} (DDL build + ANALYZE TABLE time:{" "}
                    {step4Result.ddl_time_s}s)
                  </div>
                </div>
              )}

              <div className="step-footer">
                <button
                  className="btn btn-secondary"
                  onClick={() => setStep(3)}
                >
                  &larr; Back to Step 3
                </button>
                <button
                  className="btn btn-primary"
                  onClick={() => setStep(5)}
                  disabled={!step4Result}
                >
                  Next: Step 5 (Re-Run Query With Index) &rarr;
                </button>
              </div>
            </div>
          )}

          {/* STEP 5: Re-Run Query With Index */}
          {step === 5 && (
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2 className="panel-title">
                    Step 5: Re-Run Query WITH Created Index on Live MySQL
                  </h2>
                  <p className="panel-subtitle">
                    Executes the exact same query over 10 runs (1 warm-up
                    discarded) and captures the new `EXPLAIN` and `EXPLAIN
                    ANALYZE` plans.
                  </p>
                </div>
                <button
                  className="btn btn-primary"
                  onClick={handleRunAfterIndex}
                  disabled={loading}
                >
                  {loading
                    ? "Running 10 Executions With Index..."
                    : "Run Query With Index (10 Runs, 1 Warm-Up)"}
                </button>
              </div>

              <pre className="code-box" style={{ marginBottom: 16 }}>
                {sqlText}
              </pre>

              <div
                className="alert"
                style={{
                  background: "rgba(59, 130, 246, 0.12)",
                  border: "1px solid rgba(59, 130, 246, 0.4)",
                  color: "#dbeafe",
                }}
              >
                <strong>In Simple Words — What happens in Step 3?</strong>
                <div style={{ marginTop: 4 }}>
                  Instead of guessing which column to index, our{" "}
                  <strong>Machine Learning model (XGBoost)</strong> looks at all
                  possible indexes for your query and predicts{" "}
                  <strong>how many times faster</strong> each one will be. If the
                  best index makes the query at least <strong>1.5x faster</strong>
                  , it recommends creating it; otherwise it says{" "}
                  <strong>"No Index Needed"</strong>.
                </div>
              </div>

              {step5Result && (
                <>
                  <div className="kpi-grid">
                    <div className="kpi-card">
                      <div className="kpi-label">
                        Median Execution Time (With Index)
                      </div>
                      <div className="kpi-value" style={{ color: "#34d399" }}>
                        {step5Result.measurement.median_time_ms.toFixed(2)} ms
                      </div>
                      <div className="kpi-sub">
                        Min: {step5Result.measurement.min_time_ms.toFixed(2)} ms
                        &bull; Max:{" "}
                        {step5Result.measurement.max_time_ms.toFixed(2)} ms
                      </div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-label">EXPLAIN Rows Examined After</div>
                      <div className="kpi-value" style={{ color: "#34d399" }}>
                        {Number(
                          step5Result.measurement.rows_examined
                        ).toLocaleString()}
                      </div>
                      <div className="kpi-sub">
                        Before:{" "}
                        {step2Result
                          ? Number(
                              step2Result.measurement.rows_examined
                            ).toLocaleString()
                          : "—"}
                      </div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-label">MySQL Access Type After</div>
                      <div className="kpi-value">
                        <span className="badge badge-green">
                          {step5Result.measurement.access_type}
                        </span>
                      </div>
                      <div className="kpi-sub">
                        Extra: {step5Result.measurement.extra || "Using index"}
                      </div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-label">Key Selected by Optimizer</div>
                      <div className="kpi-value" style={{ fontSize: 14 }}>
                        {step5Result.measurement.key_used || "NONE (Ignored)"}
                      </div>
                      <div className="kpi-sub">Live InnoDB Cost-Based Plan</div>
                    </div>
                  </div>

                  <div className="grid-2-equal">
                    <div>
                      <h3 style={{ fontSize: 14.5, marginBottom: 8 }}>
                        New `EXPLAIN` Plan (With Index)
                      </h3>
                      <ExplainTable rows={step5Result.measurement.explain_rows} />
                    </div>
                    <div>
                      <h3 style={{ fontSize: 14.5, marginBottom: 8 }}>
                        New `EXPLAIN ANALYZE` Tree (With Index)
                      </h3>
                      <pre className="code-box">
                        {step5Result.measurement.explain_analyze}
                      </pre>
                    </div>
                  </div>
                </>
              )}

              <div className="step-footer">
                <button
                  className="btn btn-secondary"
                  onClick={() => setStep(4)}
                >
                  &larr; Back to Step 4
                </button>
                <button
                  className="btn btn-primary"
                  onClick={handleBuildComparison}
                  disabled={!step5Result || loading}
                >
                  Next: Step 6 (Compare Predicted vs Actual &amp; Save Run) &rarr;
                </button>
              </div>
            </div>
          )}

          {/* STEP 6: Comparison Table, Bar Charts & Plain-English Diagnostic */}
          {step === 6 && (
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2 className="panel-title">
                    Step 6: Predicted vs. Actual Speedup Comparison &amp;
                    Analysis
                  </h2>
                  <p className="panel-subtitle">
                    Compares the XGBoost model's pre-index prediction against the
                    actual measured MySQL 8 speedup and saves the audit record to
                    `results/run_&lt;timestamp&gt;.json`.
                  </p>
                </div>
                <button
                  className="btn btn-danger-outline"
                  onClick={handleResetDemo}
                  disabled={loading}
                >
                  Reset Demo (Drop Created Index)
                </button>
              </div>

              <div
                className="alert"
                style={{
                  background: "rgba(59, 130, 246, 0.12)",
                  border: "1px solid rgba(59, 130, 246, 0.4)",
                  color: "#dbeafe",
                }}
              >
                <strong>In Simple Words — What happens in Step 6?</strong>
                <div style={{ marginTop: 4 }}>
                  This final screen proves our AI model was right! It compares{" "}
                  <strong>how fast the query was BEFORE the index (~900 ms)</strong>{" "}
                  vs. <strong>AFTER the index (~0.8 ms)</strong>, and compares{" "}
                  <strong>what the AI predicted</strong> against{" "}
                  <strong>what actually happened in MySQL</strong>.
                </div>
              </div>

              {!step6Result ? (
                <div className="alert alert-warning">
                  Click "Compare Predicted vs Actual" in Step 5 to populate the
                  comparison report.
                </div>
              ) : (
                <>
                  <div className="kpi-grid">
                    <div className="kpi-card">
                      <div className="kpi-label">XGBoost Predicted Speedup</div>
                      <div className="kpi-value" style={{ color: "#93c5fd" }}>
                        {step6Result.predicted_speedup.toFixed(2)}x
                      </div>
                      <div className="kpi-sub">
                        Index: {step6Result.chosen_columns || "none"}
                      </div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-label">Actual Measured Speedup</div>
                      <div className="kpi-value" style={{ color: "#34d399" }}>
                        {step6Result.actual_speedup.toFixed(2)}x
                      </div>
                      <div className="kpi-sub">
                        {step6Result.time_before_ms.toFixed(2)} ms &rarr;{" "}
                        {step6Result.time_after_ms.toFixed(2)} ms
                      </div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-label">Relative Prediction Error</div>
                      <div className="kpi-value">{step6Result.error_pct}%</div>
                      <div className="kpi-sub">
                        |Predicted - Actual| / Actual
                      </div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-label">Rows Examined Reduction</div>
                      <div className="kpi-value" style={{ color: "#34d399" }}>
                        {step6Result.row_reduction_pct}%
                      </div>
                      <div className="kpi-sub">
                        {Number(step6Result.rows_before).toLocaleString()} &rarr;{" "}
                        {Number(step6Result.rows_after).toLocaleString()} rows
                      </div>
                    </div>
                  </div>

                  {/* Comparison Table */}
                  <h3 style={{ fontSize: 15, marginBottom: 10 }}>
                    Before vs. After Empirical Comparison Table
                  </h3>
                  <div className="table-wrap" style={{ marginBottom: 18 }}>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Metric</th>
                          <th>Without Index (Step 2)</th>
                          <th>With Index (Step 5)</th>
                          <th>XGBoost Prediction (Step 3)</th>
                          <th>Measured Outcome / Delta</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>
                            <strong>Median Query Latency (10 runs)</strong>
                          </td>
                          <td className="mono">
                            {step6Result.time_before_ms.toFixed(3)} ms
                          </td>
                          <td className="mono">
                            {step6Result.time_after_ms.toFixed(3)} ms
                          </td>
                          <td className="mono">
                            {(
                              step6Result.time_before_ms /
                              Math.max(step6Result.predicted_speedup, 0.01)
                            ).toFixed(3)}{" "}
                            ms (implied)
                          </td>
                          <td className="mono">
                            <strong style={{ color: "#34d399" }}>
                              {step6Result.actual_speedup.toFixed(2)}x actual
                              speedup
                            </strong>
                          </td>
                        </tr>
                        <tr>
                          <td>
                            <strong>Speedup Ratio (t_before / t_after)</strong>
                          </td>
                          <td className="mono">1.00x (Baseline)</td>
                          <td className="mono">
                            <strong>
                              {step6Result.actual_speedup.toFixed(2)}x
                            </strong>
                          </td>
                          <td className="mono">
                            <strong>
                              {step6Result.predicted_speedup.toFixed(2)}x
                            </strong>
                          </td>
                          <td className="mono">
                            Error: {step6Result.error_pct}%
                          </td>
                        </tr>
                        <tr>
                          <td>
                            <strong>EXPLAIN Rows Examined</strong>
                          </td>
                          <td className="mono">
                            {Number(step6Result.rows_before).toLocaleString()}
                          </td>
                          <td className="mono">
                            {Number(step6Result.rows_after).toLocaleString()}
                          </td>
                          <td className="mono">—</td>
                          <td className="mono">
                            -{step6Result.row_reduction_pct}% scanned rows
                          </td>
                        </tr>
                        <tr>
                          <td>
                            <strong>MySQL Optimizer Access Path</strong>
                          </td>
                          <td className="mono">
                            type={step6Result.access_type_before}, key=NULL
                          </td>
                          <td className="mono">
                            type={step6Result.access_type_after}, key=
                            {step6Result.key_used_after || "NULL"}
                          </td>
                          <td className="mono">B-Tree Index Lookup</td>
                          <td className="mono">
                            {step6Result.key_used_after
                              ? "Optimizer adopted index"
                              : "Optimizer bypassed index"}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  {/* Interactive Visual Bar Charts */}
                  <div className="grid-2-equal" style={{ marginBottom: 18 }}>
                    <div
                      className="panel"
                      style={{ background: "var(--bg-card-alt)", marginBottom: 0 }}
                    >
                      <h4 style={{ fontSize: 14.5, marginBottom: 14 }}>
                        Bar Chart: Predicted vs. Actual Speedup (x)
                      </h4>
                      <ComparisonBarChart
                        items={[
                          {
                            label: "XGBoost Predicted Speedup",
                            value: step6Result.predicted_speedup,
                            suffix: "x",
                            color: "#3b82f6",
                          },
                          {
                            label: "Actual Measured Speedup",
                            value: step6Result.actual_speedup,
                            suffix: "x",
                            color: "#10b981",
                          },
                        ]}
                      />
                    </div>
                    <div
                      className="panel"
                      style={{ background: "var(--bg-card-alt)", marginBottom: 0 }}
                    >
                      <h4 style={{ fontSize: 14.5, marginBottom: 14 }}>
                        Bar Chart: Execution Time Before vs. After (ms)
                      </h4>
                      <ComparisonBarChart
                        items={[
                          {
                            label: "Before Index (Full Table Scan)",
                            value: step6Result.time_before_ms,
                            suffix: " ms",
                            color: "#ef4444",
                          },
                          {
                            label: "After Index (With Candidate Index)",
                            value: step6Result.time_after_ms,
                            suffix: " ms",
                            color: "#10b981",
                          },
                        ]}
                      />
                    </div>
                  </div>

                  {/* Plain-English Diagnostic Explanation */}
                  <div
                    className="panel"
                    style={{
                      background: "rgba(59, 130, 246, 0.08)",
                      borderColor: "rgba(59, 130, 246, 0.35)",
                      marginBottom: 12,
                    }}
                  >
                    <h4 style={{ fontSize: 15, marginBottom: 8, color: "#93c5fd" }}>
                      Why Prediction and Reality Differ (Diagnostic Explanation)
                    </h4>
                    <div
                      style={{
                        whiteSpace: "pre-line",
                        fontSize: 13.5,
                        color: "var(--text-main)",
                      }}
                    >
                      {step6Result.explanation}
                    </div>
                  </div>

                  <div className="alert alert-success" style={{ marginBottom: 0 }}>
                    Saved complete empirical run audit log to{" "}
                    <code className="mono">{step6Result.saved_file}</code>
                  </div>
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function ExplainTable({ rows }) {
  if (!rows || !rows.length) {
    return <div className="code-box">No EXPLAIN rows returned.</div>;
  }
  const cols = ["id", "select_type", "table", "type", "possible_keys", "key", "key_len", "rows", "filtered", "Extra"];
  const presentCols = cols.filter((c) => c in rows[0]);
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {presentCols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {presentCols.map((c) => (
                <td key={c} className="mono">
                  {r[c] === null || r[c] === undefined ? "NULL" : String(r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ComparisonBarChart({ items }) {
  const maxVal = Math.max(...items.map((i) => Number(i.value) || 0), 0.001);
  return (
    <div>
      {items.map((item) => {
        const pct = Math.max(3, Math.min(100, (item.value / maxVal) * 100));
        return (
          <div key={item.label} style={{ marginBottom: 16 }}>
            <div className="bar-meta">
              <span>{item.label}</span>
              <strong className="mono">
                {Number(item.value).toFixed(2)}
                {item.suffix}
              </strong>
            </div>
            <div className="bar-track" style={{ height: 18 }}>
              <div
                className="bar-fill"
                style={{
                  width: `${pct}%`,
                  background: item.color,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function EvaluationLab({ evalSummary, loading, onRefresh }) {
  if (loading && !evalSummary) {
    return <div className="panel">Loading held-out template evaluation...</div>;
  }
  if (!evalSummary) {
    return (
      <div className="panel">
        <button className="btn btn-primary" onClick={onRefresh}>
          Load Evaluation &amp; SHAP Report
        </button>
      </div>
    );
  }

  const metrics = evalSummary.model_metadata?.metrics || {};
  return (
    <div>
      <div className="panel">
        <div className="panel-header">
          <div>
            <h2 className="panel-title">
              Held-Out Query Template Evaluation &amp; Baseline Comparison
            </h2>
            <p className="panel-subtitle">
              Evaluated on 5 unseen SQL query templates (`T03`, `T07`, `T11`,
              `T16`, `T19` — 75 queries, 255 candidate pairs) strictly held out
              from the 16 training templates.
            </p>
          </div>
          <button className="btn btn-secondary" onClick={onRefresh}>
            Refresh Metrics
          </button>
        </div>

        <div className="kpi-grid">
          <div className="kpi-card">
            <div className="kpi-label">Held-Out MAE (log speedup)</div>
            <div className="kpi-value">{metrics.heldout_mae_log}</div>
            <div className="kpi-sub">vs 3.7428 (Legacy Rule-Based)</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Held-Out Spearman &rho;</div>
            <div className="kpi-value" style={{ color: "#34d399" }}>
              {metrics.heldout_spearman}
            </div>
            <div className="kpi-sub">Rank correlation on unseen templates</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Top-1 Exact / Effective Acc</div>
            <div className="kpi-value">
              {(metrics.top1_exact_accuracy * 100).toFixed(1)}% /{" "}
              {(metrics.top1_effective_accuracy * 100).toFixed(0)}%
            </div>
            <div className="kpi-sub">
              Index vs No-Index Decision:{" "}
              {(metrics.index_decision_accuracy * 100).toFixed(0)}%
            </div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Geometric Mean Speedup</div>
            <div className="kpi-value" style={{ color: "#34d399" }}>
              {metrics.geometric_mean_speedup}x
            </div>
            <div className="kpi-sub">
              Oracle Optimal: {metrics.oracle_geometric_mean_speedup}x
            </div>
          </div>
        </div>

        <h3 style={{ fontSize: 15, marginBottom: 10 }}>
          Baseline Comparison Table (`evaluation/results.md`)
        </h3>
        <div className="table-wrap" style={{ marginBottom: 20 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>MAE (log speedup) &darr;</th>
                <th>RMSE (log speedup) &darr;</th>
                <th>Spearman &rho; &uarr;</th>
                <th>Top-1 Exact Acc &uarr;</th>
                <th>Top-1 Effective Acc &uarr;</th>
                <th>Decision Acc &uarr;</th>
                <th>Geo-Mean Speedup &uarr;</th>
              </tr>
            </thead>
            <tbody>
              {evalSummary.baseline_comparison.map((row) => (
                <tr
                  key={row.method}
                  className={
                    row.method.includes("XGBoost") ? "highlight-row" : ""
                  }
                >
                  <td>
                    <strong>{row.method}</strong>
                  </td>
                  <td className="mono">{row.mae_log.toFixed(4)}</td>
                  <td className="mono">{row.rmse_log.toFixed(4)}</td>
                  <td className="mono">{row.spearman_rho.toFixed(4)}</td>
                  <td className="mono">
                    {(row.top1_exact_acc * 100).toFixed(1)}%
                  </td>
                  <td className="mono">
                    {(row.top1_effective_acc * 100).toFixed(1)}%
                  </td>
                  <td className="mono">
                    {(row.decision_acc * 100).toFixed(1)}%
                  </td>
                  <td className="mono">
                    <strong>{row.geo_mean_speedup.toFixed(2)}x</strong>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="grid-2">
          <div>
            <h3 style={{ fontSize: 15, marginBottom: 10 }}>
              SHAP Summary Beeswarm Plot (`evaluation/shap_summary.png`)
            </h3>
            <div
              style={{
                background: "#fff",
                borderRadius: 12,
                padding: 10,
                textAlign: "center",
              }}
            >
              <img
                src="/api/shap-plot"
                alt="SHAP Summary Plot"
                style={{ maxWidth: "100%", height: "auto", borderRadius: 8 }}
              />
            </div>
          </div>

          <div>
            <h3 style={{ fontSize: 15, marginBottom: 10 }}>
              Per-Template Breakdown on Held-Out Templates
            </h3>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Template</th>
                    <th>Group</th>
                    <th>XGB MAE</th>
                    <th>Rule MAE</th>
                    <th>XGB Speedup</th>
                  </tr>
                </thead>
                <tbody>
                  {evalSummary.template_breakdown.map((t) => (
                    <tr key={t.template_id}>
                      <td className="mono">
                        <strong>{t.template_id}</strong>
                      </td>
                      <td>
                        <span className="badge badge-blue">{t.split_group}</span>
                      </td>
                      <td className="mono">{t.xgb_mae.toFixed(3)}</td>
                      <td className="mono">{t.rule_mae.toFixed(3)}</td>
                      <td className="mono">
                        <strong>{t.xgb_geo_speedup.toFixed(1)}x</strong>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
