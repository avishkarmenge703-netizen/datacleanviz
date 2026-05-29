"""
requirements.txt
----------------
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.9.0
scipy>=1.7.0
missingno>=0.5.0
chardet>=4.0.0
streamlit>=1.20.0
"""

import os
import warnings
import base64
from io import BytesIO
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime

import chardet
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import missingno as msno

# For interactive web app
import streamlit as st

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
warnings.filterwarnings("ignore", category=FutureWarning)


class DataCleanViz:
    """
    Automated data cleaning and comprehensive data visualization for structured datasets.

    This class loads data (CSV/Excel), performs configurable cleaning,
    generates an EDA dashboard, and produces detailed reports.

    Attributes:
        filepath (str): Path to the input file.
        df (pd.DataFrame): Raw loaded DataFrame.
        cleaned_df (pd.DataFrame): Cleaned DataFrame after processing.
        cleaning_log (list): Detailed log of cleaning operations.
        report_str (str): Summary report text.
    """

    def __init__(self):
        self.filepath: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.cleaning_log: List[Dict[str, Any]] = []
        self.report_str: str = ""

    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet."""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            if encoding is None:
                encoding = 'utf-8'
        return encoding

    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load dataset from CSV or Excel with encoding detection.

        Args:
            filepath: Path to the data file.
            **kwargs: Additional arguments passed to pandas read function.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is not supported or DataFrame is empty.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        self.filepath = filepath
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.csv':
            encoding = self._detect_encoding(filepath)
            try:
                df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            except UnicodeDecodeError:
                # fallback if detection fails
                df = pd.read_csv(filepath, encoding='latin1', **kwargs)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: .csv, .xls, .xlsx")

        if df.empty:
            raise ValueError("The dataset is empty.")

        self.df = df.copy()
        self.cleaned_df = df.copy()
        self.cleaning_log = []
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df

    def _get_numeric_cols(self, df: pd.DataFrame) -> List[str]:
        """Return list of numeric column names."""
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def _get_categorical_cols(self, df: pd.DataFrame) -> List[str]:
        """Return list of categorical/object column names."""
        return df.select_dtypes(include=['object', 'category']).columns.tolist()

    def _is_normal(self, data: pd.Series, alpha: float = 0.05) -> bool:
        """Check normality using Shapiro-Wilk test (for sample sizes ≤ 5000)."""
        clean_data = data.dropna()
        if len(clean_data) < 3 or len(clean_data) > 5000:
            return False
        _, p = stats.shapiro(clean_data)
        return p > alpha

    def clean_data(self, strategy_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Perform data cleaning according to configurable strategies.

        Args:
            strategy_config: Dictionary with keys:
                - 'numeric_fill': 'median' (default), 'mean', or custom value.
                - 'categorical_fill': 'mode' (default) or 'Unknown'.
                - 'outlier_method': 'iqr' (default) or 'zscore'.
                - 'outlier_threshold': multiplier for IQR (default 1.5) or Z-score (default 3).
                - 'drop_missing_threshold': fraction above which columns are dropped (default 0.5).
                - 'capitalize_names': list of columns to title case (auto-detect if None).
                - 'cat_lowercase': list of columns to lowercase (auto-detect if None).
                - 'date_cols': list of columns to parse as dates (auto-detect if None).

        Returns:
            Cleaned DataFrame.
        """
        if self.df is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        # default strategies
        config = {
            'numeric_fill': 'median',
            'categorical_fill': 'mode',
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'drop_missing_threshold': 0.5,
            'capitalize_names': None,
            'cat_lowercase': None,
            'date_cols': None
        }
        if strategy_config:
            config.update(strategy_config)

        # Store initial state
        initial_shape = self.df.shape
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
        dup_count = self.df.duplicated().sum()

        self.cleaning_log.append({
            'operation': 'initial_state',
            'shape': initial_shape,
            'missing_pct': missing_pct.to_dict(),
            'duplicate_count': dup_count
        })

        df = self.df.copy()

        # 1. Drop columns with > threshold missing values
        threshold = config['drop_missing_threshold']
        missing_frac = df.isnull().mean()
        cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            self.cleaning_log.append({
                'operation': 'drop_high_missing_columns',
                'columns': cols_to_drop,
                'threshold': threshold
            })

        # 2. Convert date columns
        date_cols = config['date_cols']
        if date_cols is None:
            # Auto-detect potential date columns
            date_cols = [col for col in df.columns if df[col].dtype == 'object' and
                         df[col].str.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}').any()]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                self.cleaning_log.append({'operation': 'convert_to_datetime', 'column': col})
            except Exception:
                pass

        # 3. Strip whitespace from string columns
        str_cols = df.select_dtypes(include=['object']).columns
        for col in str_cols:
            df[col] = df[col].str.strip()

        # 4. Fix capitalization
        name_cols = config['capitalize_names']
        if name_cols is None:
            name_cols = [col for col in str_cols if 'name' in col.lower() or 'first' in col.lower()
                         or 'last' in col.lower()]
        for col in name_cols:
            if col in df.columns:
                df[col] = df[col].str.title()

        cat_cols = config['cat_lowercase']
        if cat_cols is None:
            cat_cols = [col for col in str_cols if col not in name_cols and df[col].nunique() < 20]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].str.lower()

        # 5. Handle missing values
        numeric_cols = self._get_numeric_cols(df)
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                fill_val = config['numeric_fill']
                if fill_val == 'median':
                    val = df[col].median()
                elif fill_val == 'mean':
                    val = df[col].mean()
                else:
                    val = fill_val  # custom value
                df[col].fillna(val, inplace=True)
                self.cleaning_log.append({
                    'operation': 'fill_na_numeric',
                    'column': col,
                    'strategy': fill_val,
                    'fill_value': val,
                    'missing_count': df[col].isnull().sum()
                })

        categorical_cols = self._get_categorical_cols(df)
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                fill_val = config['categorical_fill']
                if fill_val == 'mode':
                    val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                else:
                    val = 'Unknown'
                df[col].fillna(val, inplace=True)
                self.cleaning_log.append({
                    'operation': 'fill_na_categorical',
                    'column': col,
                    'strategy': fill_val,
                    'fill_value': val,
                    'missing_count': df[col].isnull().sum()
                })

        # 6. Remove duplicates
        dup_before = df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        dup_after = df.duplicated().sum()
        if dup_before > 0:
            self.cleaning_log.append({
                'operation': 'remove_duplicates',
                'removed_count': dup_before - dup_after
            })

        # 7. Outlier detection and capping
        outlier_method = config['outlier_method']
        threshold_out = config['outlier_threshold']
        for col in numeric_cols:
            if outlier_method == 'zscore':
                if self._is_normal(df[col]):
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outliers = z_scores > threshold_out
                    count = outliers.sum()
                    if count > 0:
                        mean = df[col].mean()
                        std = df[col].std()
                        lower = mean - threshold_out * std
                        upper = mean + threshold_out * std
                        df[col] = df[col].clip(lower, upper)
                        self.cleaning_log.append({
                            'operation': 'cap_outliers_zscore',
                            'column': col,
                            'count': count,
                            'lower': lower,
                            'upper': upper
                        })
            else:  # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold_out * IQR
                upper = Q3 + threshold_out * IQR
                outliers = (df[col] < lower) | (df[col] > upper)
                count = outliers.sum()
                if count > 0:
                    df[col] = df[col].clip(lower, upper)
                    self.cleaning_log.append({
                        'operation': 'cap_outliers_iqr',
                        'column': col,
                        'count': count,
                        'lower': lower,
                        'upper': upper
                    })

        final_shape = df.shape
        self.cleaning_log.append({
            'operation': 'final_state',
            'shape': final_shape
        })

        self.cleaned_df = df.copy()
        print(f"Data cleaning completed. New shape: {final_shape} (original: {initial_shape})")
        return self.cleaned_df

    def _generate_plots(self, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate all EDA plots and return as dictionary of matplotlib/plotly figures.

        Returns:
            Dictionary with keys like 'missing_matrix', 'histograms', etc.
        """
        if self.cleaned_df is None:
            raise RuntimeError("No cleaned data. Call clean_data() first.")

        df = self.cleaned_df.copy()
        numeric_cols = self._get_numeric_cols(df)
        categorical_cols = self._get_categorical_cols(df)

        plots = {}

        # 1. Missing data matrix (original data)
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.matrix(self.df if self.df is not None else df, ax=ax, sparkline=False)
        ax.set_title("Missing Data Matrix (Original Data)", fontsize=14)
        plots['missing_matrix'] = fig

        # 2. Univariate - Histograms + KDE for numeric
        n_cols = 3
        n_rows = max(1, -(-len(numeric_cols) // n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color=sns.color_palette("Set2")[0])
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plots['histograms'] = fig

        # 3. Univariate - Count plots for categorical
        if categorical_cols:
            n_rows_cat = max(1, -(-len(categorical_cols) // n_cols))
            fig_cat, axes_cat = plt.subplots(n_rows_cat, n_cols, figsize=(5 * n_cols, 4 * n_rows_cat))
            axes_cat = axes_cat.flatten() if isinstance(axes_cat, np.ndarray) else [axes_cat]
            for i, col in enumerate(categorical_cols):
                ax = axes_cat[i]
                value_counts = df[col].value_counts().nlargest(10)
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax,
                            palette="Set3")
                ax.set_title(f"Counts of {col}")
                ax.set_xlabel(col)
                ax.tick_params(axis='x', rotation=45)
            for j in range(i + 1, len(axes_cat)):
                axes_cat[j].set_visible(False)
            plt.tight_layout()
            plots['count_plots'] = fig_cat

        # 4. Bivariate: boxplots against target
        if target_col:
            features_num = [c for c in numeric_cols if c != target_col] if target_col in numeric_cols else numeric_cols
            if target_col in categorical_cols:
                n_rows_box = max(1, -(-len(features_num) // n_cols))
                fig_box, axes_box = plt.subplots(n_rows_box, n_cols, figsize=(5 * n_cols, 4 * n_rows_box))
                axes_box = axes_box.flatten() if isinstance(axes_box, np.ndarray) else [axes_box]
                for i, col in enumerate(features_num):
                    ax = axes_box[i]
                    sns.boxplot(x=target_col, y=col, data=df, ax=ax, palette="Set2")
                    ax.set_title(f"{col} by {target_col}")
                    ax.tick_params(axis='x', rotation=45)
                for j in range(i + 1, len(axes_box)):
                    axes_box[j].set_visible(False)
                plt.tight_layout()
                plots['boxplots_target'] = fig_box

        # 5. Correlation heatmap
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            fig_heat, ax_heat = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                        square=True, ax=ax_heat, fmt=".2f")
            ax_heat.set_title("Correlation Heatmap")
            plt.tight_layout()
            plots['correlation_heatmap'] = fig_heat

        # 6. Pairplot for top correlated numeric features
        if target_col and target_col in numeric_cols and len(numeric_cols) >= 2:
            corr_with_target = df[numeric_cols].corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
            top_feats = corr_with_target.head(4).index.tolist()
            plot_cols = top_feats + [target_col] if target_col not in top_feats else top_feats
            if len(plot_cols) > 1:
                pair_fig = sns.pairplot(df[plot_cols], diag_kind='kde', palette="Set2")
                pair_fig.fig.suptitle("Pairplot of Top Correlated Features", y=1.02)
                plots['pairplot'] = pair_fig.fig

        # 7. Clustermap
        if len(numeric_cols) > 2:
            clust_fig = sns.clustermap(df[numeric_cols].corr(), annot=True, cmap='coolwarm',
                                       center=0, figsize=(10, 8))
            clust_fig.fig.suptitle("Correlation Clustermap", y=1.02)
            plots['clustermap'] = clust_fig.fig

        # 8. Stacked bar chart
        if len(categorical_cols) >= 2:
            cat1 = categorical_cols[0]
            cat2 = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0]
            if cat1 != cat2:
                cross_tab = pd.crosstab(df[cat1], df[cat2], normalize='index')
                ax_stacked = cross_tab.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')
                ax_stacked.set_title(f"Stacked Bar: {cat1} vs {cat2}")
                ax_stacked.set_xlabel(cat1)
                ax_stacked.set_ylabel("Proportion")
                plt.legend(title=cat2, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plots['stacked_bar'] = ax_stacked.figure

        return plots

    def visualize(self, target_col: Optional[str] = None, output_format: str = 'html') -> None:
        """
        Generate and save EDA visualizations.

        Args:
            target_col: Name of target variable (optional).
            output_format: 'html' for interactive dashboard, 'png' for static images.
        """
        if self.cleaned_df is None:
            raise RuntimeError("No cleaned data. Call clean_data() first.")
        os.makedirs("visualizations", exist_ok=True)
        plots = self._generate_plots(target_col)
        if output_format == 'html':
            self._save_plots_as_png(plots)
            self._build_html_report(plots)
        else:
            self._save_plots_as_png(plots)
        print(f"Visualizations saved in 'visualizations/' directory.")

    def _save_plots_as_png(self, plots: Dict[str, Any]) -> None:
        for name, fig in plots.items():
            if isinstance(fig, plt.Figure):
                fig.savefig(f"visualizations/{name}.png", bbox_inches='tight', dpi=150)
                plt.close(fig)
            elif hasattr(fig, 'savefig'):
                fig.savefig(f"visualizations/{name}.png", bbox_inches='tight', dpi=150)
                if hasattr(fig, 'figure'):
                    plt.close(fig.figure)

    def _build_html_report(self, plots: Dict[str, Any]) -> None:
        html_content = "<html><head><title>EDA Report</title>"
        html_content += '<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css">'
        html_content += "<style>body{padding:20px;}</style></head><body>"
        html_content += "<h1>Exploratory Data Analysis Report</h1>"
        html_content += f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"

        for name in plots.keys():
            img_path = f"visualizations/{name}.png"
            if os.path.exists(img_path):
                with open(img_path, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode()
                    html_content += f"<h2>{name.replace('_',' ').title()}</h2>"
                    html_content += f'<img src="data:image/png;base64,{b64_string}" class="img-fluid" />'
                    html_content += "<hr>"
        html_content += "</body></html>"
        with open("EDA_report.html", "w") as f:
            f.write(html_content)
        print("EDA_report.html created.")

    def generate_report(self) -> str:
        """Generate a text report summarizing cleaning operations."""
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("DATA CLEANING REPORT")
        report_lines.append("=" * 50)
        for entry in self.cleaning_log:
            op = entry['operation']
            if op == 'initial_state':
                report_lines.append(f"Initial shape: {entry['shape']}")
                report_lines.append(f"Duplicate rows: {entry['duplicate_count']}")
                report_lines.append("Missing percentages:")
                for col, pct in entry['missing_pct'].items():
                    if pct > 0:
                        report_lines.append(f"  - {col}: {pct:.2f}%")
            elif op == 'drop_high_missing_columns':
                report_lines.append(f"Dropped columns (missing > {entry['threshold']*100}%): {entry['columns']}")
            elif op == 'remove_duplicates':
                report_lines.append(f"Removed {entry['removed_count']} duplicate rows")
            elif op == 'convert_to_datetime':
                report_lines.append(f"Converted column to datetime: {entry['column']}")
            elif 'fill_na' in op:
                report_lines.append(f"Filled missing in {entry['column']} using {entry['strategy']} ({entry['fill_value']})")
            elif 'cap_outliers' in op:
                report_lines.append(f"Capped outliers in {entry['column']} ({entry['count']} values) using {op}")
            elif op == 'final_state':
                report_lines.append(f"Final shape: {entry['shape']}")
        report_lines.append("=" * 50)
        self.report_str = "\n".join(report_lines)
        print(self.report_str)
        return self.report_str


def run_streamlit_app():
    """
    Streamlit web interface for DataCleanViz.
    Allows uploading data, configuring cleaning, and viewing EDA.
    """
    st.set_page_config(page_title="DataCleanViz", layout="wide")
    st.title("🧹 DataCleanViz - Automated Data Cleaning & EDA")
    st.markdown("Upload your structured dataset (CSV/Excel) and get a cleaned version with interactive visualizations.")

    if 'cleaner' not in st.session_state:
        st.session_state.cleaner = DataCleanViz()
    cleaner = st.session_state.cleaner

    with st.sidebar:
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            temp_path = os.path.join("temp_data", uploaded_file.name)
            os.makedirs("temp_data", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                df = cleaner.load_data(temp_path)
                st.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return

        st.header("⚙️ Cleaning Options")
        numeric_fill = st.selectbox("Numeric missing fill", ["median", "mean", "custom"], index=0)
        custom_fill_val = None
        if numeric_fill == "custom":
            custom_fill_val = st.number_input("Custom fill value", value=0.0)
            numeric_fill = custom_fill_val

        categorical_fill = st.selectbox("Categorical missing fill", ["mode", "Unknown"], index=0)
        outlier_method = st.selectbox("Outlier detection", ["iqr", "zscore"], index=0)
        outlier_thresh = st.number_input("Threshold multiplier", value=1.5 if outlier_method == "iqr" else 3.0, step=0.1)
        drop_thresh = st.slider("Drop columns with missing % > ", 0.0, 1.0, 0.5)

        if st.button("Run Cleaning"):
            config = {
                'numeric_fill': numeric_fill,
                'categorical_fill': categorical_fill,
                'outlier_method': outlier_method,
                'outlier_threshold': outlier_thresh,
                'drop_missing_threshold': drop_thresh
            }
            with st.spinner("Cleaning data..."):
                cleaner.clean_data(strategy_config=config)
            st.success("Data cleaned!")
            with st.expander("Cleaning Report"):
                report = cleaner.generate_report()
                st.text(report)

    if cleaner.cleaned_df is not None:
        tab1, tab2, tab3 = st.tabs(["Cleaned Data", "EDA Plots", "Raw Data"])
        with tab1:
            st.subheader("Cleaned Data Preview")
            st.dataframe(cleaner.cleaned_df.head(100))
            st.download_button(
                label="Download Cleaned CSV",
                data=cleaner.cleaned_df.to_csv(index=False).encode('utf-8'),
                file_name='cleaned_data.csv',
                mime='text/csv'
            )
        with tab2:
            st.subheader("Exploratory Data Analysis")
            target_col = st.selectbox("Target variable (optional)", options=[None] + list(cleaner.cleaned_df.columns))
            if st.button("Generate EDA"):
                with st.spinner("Generating plots..."):
                    plots = cleaner._generate_plots(target_col)
                    for name, fig in plots.items():
                        st.markdown(f"### {name.replace('_', ' ').title()}")
                        # Handle both matplotlib and plotly figures
                        if isinstance(fig, go.Figure):
                            st.plotly_chart(fig, use_container_width=True)
                        elif isinstance(fig, plt.Figure):
                            st.pyplot(fig)
                        else:
                            st.pyplot(fig)  # fallback for clustermap etc.
        with tab3:
            if cleaner.df is not None:
                st.subheader("Original Data")
                st.dataframe(cleaner.df.head(100))
    else:
        st.info("Please upload a file and run cleaning from the sidebar.")


def main():
    """Console demo: creates a sample dataset, cleans it, and generates EDA."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'Name': ['John Doe', 'Jane Smith', 'Bob Brown', 'Alice Johnson', np.nan] * (n // 5),
        'Age': np.random.normal(40, 10, n).astype(int),
        'Salary': np.random.normal(70000, 15000, n),
        'Department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], n),
        'Start_Date': pd.date_range('2015-01-01', periods=n, freq='W').strftime('%Y-%m-%d').tolist()
    })
    df.loc[0, 'Age'] = np.nan
    df.loc[1, 'Salary'] = 2000000
    df.loc[2, 'Department'] = np.nan
    df.to_csv('employee_data.csv', index=False)
    print("Created employee_data.csv")
    cleaner = DataCleanViz()
    cleaner.load_data("employee_data.csv")
    cleaner.clean_data()
    cleaner.visualize(target_col="Salary", output_format="html")
    cleaner.generate_report()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "console":
        # Console demo: python script.py console
        main()
    else:
        # Streamlit mode: streamlit run script.py
        run_streamlit_app()
