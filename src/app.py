import os
import io
import base64
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
import matplotlib

# Use non-GUI backend for servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Configuration
ALLOWED_EXTENSIONS = {"csv", "xls", "xlsx"}
# Vercel has 4.5MB payload limit, so we'll use 4MB to be safe
MAX_CONTENT_LENGTH_MB = 4

# In-memory cache for last computed stats (for CSV export)
LAST_STATS_CSV: pd.DataFrame | None = None


def create_app() -> Flask:
	app = Flask(__name__)

	# Secret key for flashing messages (override in production)
	app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

	# Uploads - Use temp directory for Vercel compatibility
	import tempfile
	app.config["UPLOAD_FOLDER"] = tempfile.gettempdir()
	app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024

	@app.route("/")
	def index():
		return render_template("index.html")

	@app.route("/analyze", methods=["POST"])
	def analyze():
		global LAST_STATS_CSV
		if "dataset" not in request.files:
			flash("No file part in the request.", "error")
			return redirect(url_for("index"))

		file = request.files["dataset"]
		if file.filename == "":
			flash("No file selected.", "error")
			return redirect(url_for("index"))

		if not allowed_file(file.filename):
			flash("Invalid file type. Please upload a CSV or Excel file.", "error")
			return redirect(url_for("index"))

		filename = secure_filename(file.filename)
		# Use temporary file for Vercel compatibility
		import tempfile
		temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
		file_path = temp_file.name
		temp_file.close()
		
		try:
			file.save(file_path)
		except Exception as e:
			flash(f"Failed to save file: {e}", "error")
			return redirect(url_for("index"))

		try:
			df = load_dataframe(file_path)
		except Exception as e:
			flash(f"Failed to read file: {e}", "error")
			return redirect(url_for("index"))

		if df.empty:
			flash("Uploaded dataset is empty.", "error")
			return redirect(url_for("index"))

		# Basic numeric stats
		numeric_df = df.select_dtypes(include=[np.number])
		stats = compute_basic_stats(numeric_df)

		# Advanced EDA: preview, dtypes, missing values, duplicates
		preview_html = df.head(10).to_html(classes="striped", index=False)
		dtypes_rows = [{"column": c, "dtype": str(t)} for c, t in df.dtypes.items()]
		missing_rows = []
		for c in df.columns:
			missing = int(df[c].isna().sum())
			pct = (missing / len(df)) * 100 if len(df) else 0
			missing_rows.append({"column": c, "missing": missing, "missing_pct": f"{pct:.1f}%"})
		duplicates_count = int(df.duplicated().sum())

		# Visualizations with captions (extended)
		images = []
		try:
			images.extend(generate_visualizations(df, numeric_df))
		except Exception as e:
			# Do not fail the whole request if charts fail
			flash(f"Visualization error: {e}", "warning")

		# AI Summary
		try:
			summary = generate_ai_summary(df, stats)
		except Exception as e:
			summary = f"Automated summary unavailable due to an error: {e}"

		# Prepare stats for template rendering and export
		stats_rows = []
		for col, s in stats.items():
			stats_rows.append({
				"column": col,
				"mean": fmt_float(s.get("mean")),
				"median": fmt_float(s.get("median")),
				"mode": fmt_float(s.get("mode")),
				"std": fmt_float(s.get("std")),
			})

		# Cache CSV for download
		LAST_STATS_CSV = pd.DataFrame(stats_rows)

		# Clean up temporary file
		try:
			os.unlink(file_path)
		except:
			pass  # Ignore cleanup errors

		return render_template(
			"result.html",
			filename=filename,
			uploaded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
			stats_rows=stats_rows,
			images=images,
			summary=summary,
			preview_html=preview_html,
			dtypes_rows=dtypes_rows,
			missing_rows=missing_rows,
			duplicates_count=duplicates_count,
		)

	@app.route("/download_stats")
	def download_stats():
		global LAST_STATS_CSV
		if LAST_STATS_CSV is None or LAST_STATS_CSV.empty:
			flash("No stats available to download. Please analyze a dataset first.", "warning")
			return redirect(url_for("index"))
		buffer = io.BytesIO()
		LAST_STATS_CSV.to_csv(buffer, index=False)
		buffer.seek(0)
		return send_file(
			buffer,
			as_attachment=True,
			mimetype="text/csv",
			download_name="stats_summary.csv",
		)

	return app


def allowed_file(filename: str) -> bool:
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_dataframe(path: str) -> pd.DataFrame:
	ext = os.path.splitext(path)[1].lower()
	if ext == ".csv":
		return pd.read_csv(path)
	if ext in {".xls", ".xlsx"}:
		return pd.read_excel(path)
	raise ValueError("Unsupported file type")


def compute_basic_stats(numeric_df: pd.DataFrame) -> dict:
	stats: dict[str, dict[str, float]] = {}
	if numeric_df is None or numeric_df.empty:
		return stats
	for column in numeric_df.columns:
		series = numeric_df[column].dropna()
		if series.empty:
			continue
		
		# Check if the series is actually numeric
		if not pd.api.types.is_numeric_dtype(series):
			continue
			
		try:
			col_mode = series.mode().iloc[0] if not series.mode().empty else np.nan
		except Exception:
			col_mode = np.nan
			
		stats[column] = {
			"mean": float(series.mean()) if len(series) else np.nan,
			"median": float(series.median()) if len(series) else np.nan,
			"mode": float(col_mode) if pd.api.types.is_numeric_dtype(series) else np.nan,
			"std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
		}
		# Replace inf/-inf with nan
		for k, v in list(stats[column].items()):
			if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
				stats[column][k] = np.nan
	return stats


def fig_to_base64(figure: plt.Figure) -> str:
	buffer = io.BytesIO()
	figure.tight_layout()
	figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
	plt.close(figure)
	buffer.seek(0)
	encoded = base64.b64encode(buffer.read()).decode("utf-8")
	return f"data:image/png;base64,{encoded}"


def generate_visualizations(df: pd.DataFrame, numeric_df: pd.DataFrame) -> list[dict]:
	images: list[dict] = []

	# 1) Bar chart: mean per numeric column
	if numeric_df is not None and not numeric_df.empty:
		means = numeric_df.mean(numeric_only=True)
		if not means.empty:
			fig, ax = plt.subplots(figsize=(8, 4))
			means.plot(kind="bar", ax=ax, color="#4C78A8")
			ax.set_title("Mean per Numeric Column")
			ax.set_ylabel("Mean")
			ax.set_xlabel("Column")
			ax.grid(axis="y", linestyle=":", alpha=0.5)
			images.append({"img": fig_to_base64(fig), "caption": "Mean per numeric column", "x": "Column", "y": "Mean"})

	# 2) Line chart: first numeric column over row index
	if numeric_df is not None and not numeric_df.empty:
		first_num_col = numeric_df.columns[0]
		fig, ax = plt.subplots(figsize=(8, 4))
		ax.plot(numeric_df.index, numeric_df[first_num_col], color="#F58518")
		ax.set_title(f"Line Chart of {first_num_col} over Rows")
		ax.set_xlabel("Row Index")
		ax.set_ylabel(first_num_col)
		ax.grid(True, linestyle=":", alpha=0.5)
		images.append({"img": fig_to_base64(fig), "caption": f"Trend of {first_num_col} over rows", "x": "Row Index", "y": first_num_col})

	# 3) Pie chart: distribution of top categories in first non-numeric column
	non_numeric_df = df.select_dtypes(exclude=[np.number])
	if non_numeric_df is not None and not non_numeric_df.empty:
		cat_col = non_numeric_df.columns[0]
		counts = non_numeric_df[cat_col].astype(str).value_counts().head(6)
		if not counts.empty:
			fig, ax = plt.subplots(figsize=(6, 6))
			ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
			ax.set_title(f"Distribution of {cat_col} (Top 6)")
			images.append({"img": fig_to_base64(fig), "caption": f"Distribution of {cat_col} (Top 6)", "x": cat_col, "y": "Percentage"})

	# 4) Correlation heatmap for numeric columns
	if numeric_df is not None and numeric_df.shape[1] >= 2:
		corr = numeric_df.corr(numeric_only=True)
		fig, ax = plt.subplots(figsize=(6, 5))
		cax = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
		ax.set_xticks(range(len(corr.columns)))
		ax.set_yticks(range(len(corr.columns)))
		ax.set_xticklabels(list(corr.columns), rotation=45, ha="right")
		ax.set_yticklabels(list(corr.columns))
		ax.set_title("Correlation Heatmap")
		fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
		images.append({"img": fig_to_base64(fig), "caption": "Correlation heatmap", "x": "Variables", "y": "Variables"})

	# 5) Histograms for up to 3 numeric columns
	if numeric_df is not None and not numeric_df.empty:
		for col in list(numeric_df.columns)[:3]:
			fig, ax = plt.subplots(figsize=(6, 4))
			numeric_df[col].dropna().hist(ax=ax, bins=30, color="#54A24B")
			ax.set_title(f"Histogram of {col}")
			ax.set_xlabel(col)
			ax.set_ylabel("Count")
			images.append({"img": fig_to_base64(fig), "caption": f"Histogram of {col}", "x": col, "y": "Count"})

	# 6) Boxplots for up to 5 numeric columns
	if numeric_df is not None and numeric_df.shape[1] >= 1:
		cols = list(numeric_df.columns)[:5]
		fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.5), 4))
		ax.boxplot([numeric_df[c].dropna().values for c in cols], tick_labels=cols, patch_artist=True)
		ax.set_title("Boxplots of numeric columns")
		ax.set_ylabel("Values")
		images.append({"img": fig_to_base64(fig), "caption": "Boxplots of numeric columns", "x": "Columns", "y": "Values"})

	# 7) Strongest correlation scatter (top absolute corr)
	if numeric_df is not None and numeric_df.shape[1] >= 2:
		corr = numeric_df.corr(numeric_only=True)
		pairs = []
		for i, c1 in enumerate(corr.columns):
			for c2 in corr.columns[i + 1:]:
				val = corr.loc[c1, c2]
				if pd.notna(val):
					pairs.append((c1, c2, abs(float(val)), float(val)))
		if pairs:
			best = sorted(pairs, key=lambda x: x[2], reverse=True)[0]
			x, y, _, r = best
			fig, ax = plt.subplots(figsize=(6, 4))
			ax.scatter(numeric_df[x], numeric_df[y], alpha=0.6, color="#e45756")
			ax.set_title(f"Scatter: {x} vs {y} (r={r:.2f})")
			ax.set_xlabel(x)
			ax.set_ylabel(y)
			images.append({"img": fig_to_base64(fig), "caption": f"Scatter: {x} vs {y} (r={r:.2f})", "x": x, "y": y})

	return images


def build_summary_prompt(stats: dict, df: pd.DataFrame) -> str:
	lines = [
		"You are a data analyst. Summarize the key insights, trends, and anomalies ",
		"based on the following dataset statistics and column values. Provide concise, ",
		"actionable insights in 5-8 bullet points. Mention potential outliers or correlations if noticeable.",
		"\n\n",
		"Dataset shape: ",
		f"rows={len(df)}, cols={len(df.columns)}\n",
		"Numeric stats:"
	]
	for col, s in stats.items():
		lines.append(
			f"- {col}: mean={fmt_float(s.get('mean'))}, median={fmt_float(s.get('median'))}, "
			f"mode={fmt_float(s.get('mode'))}, std={fmt_float(s.get('std'))}"
		)

	# Sample a few value ranges per numeric column
	numeric_df = df.select_dtypes(include=[np.number])
	for col in list(numeric_df.columns)[:5]:
		series = numeric_df[col].dropna()
		if series.empty:
			continue
		q1, q3 = series.quantile(0.25), series.quantile(0.75)
		lines.append(
			f"- Range hints for {col}: min={fmt_float(series.min())}, max={fmt_float(series.max())}, "
			f"q1={fmt_float(q1)}, q3={fmt_float(q3)}"
		)

	return "".join(lines)


def generate_ai_summary(df: pd.DataFrame, stats: dict) -> str:
	prompt = build_summary_prompt(stats, df)

	# Try OpenAI if available, otherwise use rule-based fallback
	api_key = os.environ.get("OPENAI_API_KEY")
	if api_key:
		try:
			# Use OpenAI responses API
			from openai import OpenAI

			client = OpenAI(api_key=api_key)
			completion = client.chat.completions.create(
				model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
				messages=[
					{"role": "system", "content": "You are a helpful data analyst."},
					{"role": "user", "content": prompt},
				],
				temperature=0.4,
				max_tokens=400,
			)
			return completion.choices[0].message.content.strip()
		except Exception as e:
			# Fall back if API fails
			return rule_based_summary(df, stats) + f"\n\n(Note: OpenAI call failed: {e})"
	else:
		return rule_based_summary(df, stats)


def rule_based_summary(df: pd.DataFrame, stats: dict) -> str:
	lines: list[str] = []
	lines.append("Insights (non-AI fallback):")
	if not stats:
		lines.append("- No numeric columns detected. Consider including quantitative data for analysis.")
	else:
		# Identify columns with high variability
		stds = {col: s.get("std") for col, s in stats.items() if s.get("std") is not None}
		std_sorted = sorted(stds.items(), key=lambda kv: (kv[1] is None, kv[1]), reverse=True)
		if std_sorted:
			top_var = [f"{col} (std={fmt_float(std)})" for col, std in std_sorted[:3] if std is not None]
			if top_var:
				lines.append("- Highest variability: " + ", ".join(top_var))

		# Potential outliers using IQR
		numeric_df = df.select_dtypes(include=[np.number])
		outlier_notes = []
		for col in list(numeric_df.columns)[:5]:
			series = numeric_df[col].dropna()
			if len(series) < 5:
				continue
			q1, q3 = series.quantile(0.25), series.quantile(0.75)
			iqr = q3 - q1
			lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
			count_outliers = int(((series < lower) | (series > upper)).sum())
			if count_outliers > 0:
				outlier_notes.append(f"{col}: {count_outliers} potential outliers")
		if outlier_notes:
			lines.append("- Potential outliers detected: " + ", ".join(outlier_notes))

		# Trend hint on first numeric column
		if numeric_df is not None and not numeric_df.empty:
			col = numeric_df.columns[0]
			series = numeric_df[col].dropna()
			if len(series) >= 3:
				first, last = float(series.iloc[0]), float(series.iloc[-1])
				delta = last - first
				direction = "increasing" if delta > 0 else ("decreasing" if delta < 0 else "stable")
				lines.append(f"- '{col}' appears {direction} from start to end (Δ={fmt_float(delta)}).")

		# Correlation hints
		if numeric_df is not None and numeric_df.shape[1] >= 2:
			corr = numeric_df.corr(numeric_only=True)
			candidates = []
			for i, c1 in enumerate(corr.columns):
				for c2 in corr.columns[i + 1:]:
					val = corr.loc[c1, c2]
					if pd.notna(val) and abs(val) >= 0.6:
						candidates.append((c1, c2, float(val)))
			if candidates:
				top = sorted(candidates, key=lambda x: abs(x[2]), reverse=True)[:3]
				lines.append(
					"- Strong correlations: "
					+ ", ".join([f"{a}~{b} (r={fmt_float(r)})" for a, b, r in top])
				)

	# Categorical distribution hint
	non_numeric_df = df.select_dtypes(exclude=[np.number])
	if non_numeric_df is not None and not non_numeric_df.empty:
		col = non_numeric_df.columns[0]
		top = non_numeric_df[col].astype(str).value_counts().head(3)
		if not top.empty:
			pairs = [f"{idx} ({cnt})" for idx, cnt in zip(top.index, top.values)]
			lines.append(f"- '{col}' top categories: " + ", ".join(pairs))

	return "\n".join(lines)


def fmt_float(val) -> str:
	if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
		return "NA"
	try:
		# Use .4f for 4 decimal places precision
		return f"{float(val):.4f}"
	except Exception:
		return str(val)


# Create app instance for Vercel
app = create_app()

if __name__ == "__main__":
	# Running locally: python app.py
	port = int(os.environ.get("PORT", 5000))
	app.run(host="0.0.0.0", port=port, debug=True)
