import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go

st.set_page_config(page_title="Goodreads Recap", page_icon="📚", layout="wide")

# -----------------------------
# Goodreads-ish theme constants
# -----------------------------
GR_BG = "#F4F1EA"        # warm parchment
GR_TEXT = "#382110"      # goodreads brown
GR_ACCENT = "#00635D"    # goodreads green
GR_ACCENT_2 = "#B46A2A"  # warm bronze
GR_MUTED = "#D6D0C4"     # light border

alt.themes.enable("none")
alt.renderers.set_embed_options(actions=False)

def apply_altair_theme():
    return {
        "config": {
            "background": GR_BG,
            "view": {"fill": GR_BG, "stroke": GR_MUTED, "strokeWidth": 1},
            "axis": {
                "labelColor": GR_TEXT,
                "titleColor": GR_TEXT,
                "gridColor": GR_MUTED,
                "domainColor": GR_MUTED,
                "tickColor": GR_MUTED,
            },
            "legend": {"labelColor": GR_TEXT, "titleColor": GR_TEXT},
            "title": {"color": GR_TEXT, "font": "Merriweather"},
        }
    }

alt.themes.register("goodreadsish", apply_altair_theme)
alt.themes.enable("goodreadsish")


# -----------------------------
# CSS: fonts + colors
# -----------------------------
st.markdown(
    f"""
    <style>
    /* Import fonts reminiscent of Goodreads */
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&family=Lato:wght@300;400;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Lato', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        color: {GR_TEXT};
        background: {GR_BG};
    }}

    /* Main app background */
    .stApp {{
        background: {GR_BG};
    }}

    /* Headings */
    h1, h2, h3 {{
        font-family: 'Merriweather', Georgia, serif !important;
        color: {GR_TEXT} !important;
        letter-spacing: -0.2px;
    }}

    /* Sidebar background */
    section[data-testid="stSidebar"] {{
        background: #EFEADF;
        border-right: 1px solid {GR_MUTED};
    }}

    /* Metric cards look nicer on parchment */
    div[data-testid="stMetric"] {{
        background: #FFF;
        border: 1px solid {GR_MUTED};
        border-radius: 10px;
        padding: 12px 14px;
    }}

    /* Highlight box */
    .highlight-box {{
        background: #FFF;
        border: 1px solid {GR_MUTED};
        border-left: 6px solid {GR_ACCENT};
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 10px;
    }}

    .highlight-title {{
        font-family: 'Merriweather', Georgia, serif;
        font-weight: 700;
        margin-bottom: 8px;
        color: {GR_TEXT};
    }}

    .highlight-item {{
        margin: 8px 0;
        line-height: 1.35;
    }}

    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: {GR_BG};
        border: 1px solid {GR_MUTED};
        font-size: 12px;
        margin-left: 8px;
        color: {GR_TEXT};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.replace("", pd.NA), errors="coerce")

def year_filter(dt: pd.Series, year: int) -> pd.Series:
    return dt.dt.year == year

def safe_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Float64")

def safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Float64")

def review_len(s: pd.Series) -> pd.Series:
    s2 = s.fillna("").astype(str).replace("nan", "")
    return s2.str.len()

def rating_personality(avg_delta: float, n_rated: int) -> str:
    if n_rated < 5:
        return "Not enough rated books to infer your rating style (try rating a few more)."
    if avg_delta <= -0.60:
        return "You rate lower than the crowd. High standards."
    if -0.60 < avg_delta <= -0.20:
        return "Slightly harsher than average."
    if -0.20 < avg_delta < 0.20:
        return "Your ratings line up closely with Goodreads averages."
    if 0.20 <= avg_delta < 0.60:
        return "More generous than average."
    if avg_delta >= 0.60:
        return "Very generous rater."
    return "Mixed rating style."

from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

def sanitize_for_streamlit(table: pd.DataFrame) -> pd.DataFrame:
    t = table.copy()

    # Convert datetime columns to strings (safe for Streamlit/React)
    for col in t.columns:
        if is_datetime64_any_dtype(t[col]):
            t[col] = pd.to_datetime(t[col], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")

    # Convert pandas nullable numerics (Float64/Int64) to plain float
    for col in t.columns:
        if is_numeric_dtype(t[col]):
            # This will turn pandas nullable numbers into regular floats/ints where possible
            t[col] = pd.to_numeric(t[col], errors="coerce")

    # Ensure objects are simple strings (no <NA>/nan)
    for col in t.columns:
        if t[col].dtype == "object":
            t[col] = (
                t[col]
                .astype(str)
                .replace({"nan": "", "<NA>": ""})
            )

    return t


def make_sankey(added_count: int, read_count: int, both_count: int, year: int) -> go.Figure:
    not_read = max(added_count - both_count, 0)
    added_before = max(read_count - both_count, 0)

    labels = [
        f"Added in {year}",
        f"Read in {year}",
        f"Not read in {year}",
        f"Added before {year} / unknown",
    ]

    # Brighter palette
    node_colors = [
        "#00A99D",  # bright teal
        "#2E7D32",  # vivid green
        "#F57C00",  # orange
        "#7B1FA2",  # purple
    ]

    sources = [0, 0, 3]
    targets = [1, 2, 1]
    values  = [both_count, not_read, added_before]

    link_colors = [
        "rgba(46,125,50,0.55)",   # to Read
        "rgba(245,124,0,0.55)",   # to Not read
        "rgba(123,31,162,0.55)",  # Added before -> Read
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            label=labels,
            pad=15,
            thickness=18,
            color=node_colors,
            line=dict(color="rgba(56,33,16,0.25)", width=1),
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
    )])

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        paper_bgcolor=GR_BG,
        plot_bgcolor=GR_BG,
        font=dict(color=GR_TEXT, family="Lato"),
    )
    return fig


# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("Settings")
    recap_year = st.number_input("Recap year", min_value=2000, max_value=2100, value=2025, step=1)
    YEAR = int(recap_year)
    st.divider()
    st.write("This recap uses Date Read / Date Added to determine whether a book belongs in the year.")

st.title(f"Goodreads {recap_year} Recap")

st.markdown(
    """
    <div class="highlight-box" style="border-left-color:#B46A2A;">
      <div class="highlight-title">How to use</div>
      <div class="highlight-item">
        Go to <b>My Books</b> → <b>Tools</b> (left bar) → <b>Import and Export</b> → <b>Export</b> → Download your CSV, drop it here, enjoy!
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="highlight-box" style="border-left-color:#00635D;">
      <div class="highlight-title">Author's Note</div>
      <div class="highlight-item">
        I made this jank year in review in 2 hours goodreads step up your game !!! and set up an API so I can scrape stuff!!!
        Also if you're missing books in the total it might be because I noticed some books show up with weird read date? idk sorry 
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Upload your Goodreads CSV export. The file is only used during your session and is not stored.")


uploaded = st.file_uploader("Upload Goodreads CSV", type=["csv"])


if not uploaded:
    st.stop()

# Read CSV
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error("Could not read the CSV. Make sure it’s a Goodreads export.")
    st.exception(e)
    st.stop()

# Parse & coerce
df["Date Read"] = parse_date(df.get("Date Read", pd.Series([pd.NaT] * len(df))))
df["Date Added"] = parse_date(df.get("Date Added", pd.Series([pd.NaT] * len(df))))
df["My Rating"] = safe_int(df.get("My Rating", pd.Series([pd.NA] * len(df))))
df["Average Rating"] = safe_float(df.get("Average Rating", pd.Series([pd.NA] * len(df))))
df["Number of Pages"] = safe_int(df.get("Number of Pages", pd.Series([pd.NA] * len(df))))
df["Original Publication Year"] = safe_int(df.get("Original Publication Year", pd.Series([pd.NA] * len(df))))
df["Exclusive Shelf"] = df.get("Exclusive Shelf", pd.Series([""] * len(df))).fillna("").astype(str)
df["My Review"] = df.get("My Review", pd.Series([""] * len(df))).fillna("").astype(str)

# Cohorts
read_yr = df[year_filter(df["Date Read"], recap_year)].copy()
added_yr = df[year_filter(df["Date Added"], recap_year)].copy()
in_yr = df[year_filter(df["Date Read"], recap_year) | year_filter(df["Date Added"], recap_year)].copy()

# KPIs
col1, col2, col3, col4 = st.columns(4)
books_read = len(read_yr)
books_added = len(added_yr)
pages_read = int(read_yr["Number of Pages"].fillna(0).sum())
rated_count = int((read_yr["My Rating"].fillna(0) > 0).sum())

col1.metric(f"Books read in {YEAR}", f"{books_read:,}")
col2.metric(f"Books added in {YEAR}", f"{books_added:,}")
col3.metric(f"Pages read in {YEAR}", f"{pages_read:,}")
col4.metric("Read books rated", f"{rated_count:,}")

st.divider()

# -----------------------------
# Reading by month + Sankey
# -----------------------------
left, right = st.columns([1.1, 0.9])

with left:
    st.subheader(f"Reading by month ({YEAR})")


    if books_read == 0:
        st.info(f"No books with Date Read in {YEAR} were found.")

    else:
        read_yr["Read Month"] = read_yr["Date Read"].dt.to_period("M").dt.to_timestamp()
        monthly = (
            read_yr.groupby("Read Month", dropna=True)
            .agg(books=("Book Id", "count"), pages=("Number of Pages", "sum"))
            .reset_index()
        )
        monthly["pages"] = monthly["pages"].fillna(0)

        base = alt.Chart(monthly).encode(x=alt.X("Read Month:T", title="Month"))

        books_chart = base.mark_bar(color=GR_ACCENT).encode(
            y=alt.Y("books:Q", title="Books read"),
            tooltip=["Read Month:T", "books:Q", "pages:Q"],
        )

        pages_line = base.mark_line(color=GR_ACCENT_2).encode(
            y=alt.Y("pages:Q", title="Pages read"),
            tooltip=["Read Month:T", "books:Q", "pages:Q"],
        )

        st.altair_chart(
            alt.layer(books_chart, pages_line).resolve_scale(y="independent"),
            use_container_width=True,
        )

with right:
    st.subheader(f"Added vs Read flow ({YEAR})")


    added_ids = set(added_yr.get("Book Id", pd.Series(dtype="object")).dropna().astype(str).tolist())
    read_ids = set(read_yr.get("Book Id", pd.Series(dtype="object")).dropna().astype(str).tolist())
    both = len(added_ids.intersection(read_ids))

    st.plotly_chart(make_sankey(len(added_ids), len(read_ids), both, YEAR), use_container_width=True)

st.divider()

# -----------------------------
# Book age histogram
# -----------------------------
st.subheader(f"How old were the books you read in {YEAR}?")


if books_read == 0:
    st.info(f"No read books in {YEAR} to analyze.")
else:
    tmp = read_yr.dropna(subset=["Original Publication Year"]).copy()
    tmp["Book Age"] = recap_year - tmp["Original Publication Year"]
    tmp = tmp[(tmp["Book Age"] >= 0) & (tmp["Book Age"] <= 300)]

    if len(tmp) == 0:
        st.info("No usable Original Publication Year values found for your {YEAR} reads.")
    else:
        age_hist = alt.Chart(tmp).mark_bar(color=GR_ACCENT).encode(
            x=alt.X("Book Age:Q", bin=alt.Bin(maxbins=30),
                    title=f"Age in {recap_year} (years since original publication)"),
            y=alt.Y("count():Q", title="Books"),
            tooltip=[alt.Tooltip("count():Q", title="Books")],
        )
        st.altair_chart(age_hist, use_container_width=True)

        median_age = float(tmp["Book Age"].median())
        oldest = int(tmp["Book Age"].max())
        newest = int(tmp["Book Age"].min())
        st.write(f"Median age: {median_age:.0f} years. Oldest: {oldest}. Newest: {newest}.")

st.divider()

# -----------------------------
# Ratings vs averages + Highlights box
# -----------------------------
st.subheader("Your ratings vs Goodreads average")

if books_read == 0:
    st.info("No read books in {YEAR} to analyze.")
else:
    rated = read_yr[(read_yr["My Rating"].fillna(0) > 0) & read_yr["Average Rating"].notna()].copy()
    if len(rated) == 0:
        st.info(f"No rated books with Average Rating found in your {YEAR} reads.")
    else:
        rated["Delta"] = rated["My Rating"].astype(float) - rated["Average Rating"].astype(float)
        delta_mean = float(rated["Delta"].mean())

        c1, c2, c3 = st.columns(3)
        c1.metric("Your avg rating", f"{rated['My Rating'].mean():.2f}")
        c2.metric("Goodreads avg (those books)", f"{rated['Average Rating'].mean():.2f}")
        c3.metric("You minus Goodreads", f"{delta_mean:+.2f}")

        st.write(rating_personality(delta_mean, len(rated)))

        # Compute padded domains to “center” the cloud
        x_min = float(rated["Average Rating"].min())
        x_max = float(rated["Average Rating"].max())
        y_min = float(rated["My Rating"].min())
        y_max = float(rated["My Rating"].max())

        x_pad = max((x_max - x_min) * 0.15, 0.2)
        y_pad = max((y_max - y_min) * 0.15, 0.2)

        x_dom = [x_min - x_pad, x_max + x_pad]
        y_dom = [y_min - y_pad, y_max + y_pad]

        scatter = alt.Chart(rated).mark_circle(size=70, color=GR_ACCENT).encode(
            x=alt.X(
                "Average Rating:Q",
                title="Goodreads average rating",
                scale=alt.Scale(domain=x_dom),
                axis=alt.Axis(tickCount=5)  # less granular
            ),
            y=alt.Y(
                "My Rating:Q",
                title="Your rating",
                scale=alt.Scale(domain=y_dom),
                axis=alt.Axis(tickCount=6)
            ),
            tooltip=["Title:N", "Author:N", "My Rating:Q", "Average Rating:Q", "Delta:Q"],
        ).properties(height=380).interactive()

        # Diagonal y=x reference line spanning the visible range
        line_extent = [min(x_dom[0], y_dom[0]), max(x_dom[1], y_dom[1])]
        line_df = pd.DataFrame({"x": line_extent, "y": line_extent})

        ref_line = alt.Chart(line_df).mark_line(color=GR_ACCENT_2, strokeDash=[6, 4]).encode(
            x="x:Q",
            y="y:Q"
        )

        st.altair_chart(scatter + ref_line, use_container_width=True)



        # # Scatter (no diagonal line per request)
        # scatter = alt.Chart(rated).mark_circle(size=70, color=GR_ACCENT).encode(
        #     x=alt.X("Average Rating:Q", title="Goodreads average rating"),
        #     y=alt.Y("My Rating:Q", title="Your rating"),
        #     tooltip=["Title:N", "Author:N", "My Rating:Q", "Average Rating:Q", "Delta:Q"],
        # ).interactive()
        # st.altair_chart(scatter, use_container_width=True)

        # Highlights:
        # 1) largest negative delta (you below avg)
        # 2) largest positive delta (you above avg)
        # 3) lowest Average Rating among books you read (regardless of whether you rated)
        neg = rated.sort_values("Delta", ascending=True).iloc[0]
        pos = rated.sort_values("Delta", ascending=False).iloc[0]

        read_with_avg = read_yr[read_yr["Average Rating"].notna()].copy()
        if len(read_with_avg) > 0:
            underdog = read_with_avg.sort_values("Average Rating", ascending=True).iloc[0]
        else:
            underdog = None
        h1, h2, h3 = st.columns(3)

        def card_html(title, subtitle, badge, note, accent):
            return f"""
            <div class="highlight-box" style="border-left-color:{accent}; height: 100%;">
            <div class="highlight-title">{title}</div>
            <div class="highlight-item"><b>{subtitle}</b></div>
            <div class="highlight-item">{badge}</div>
            <div class="highlight-item">{note}</div>
            </div>
            """

        def fmt_book(row):
            title = str(row.get("Title", "(unknown)"))
            author = str(row.get("Author", "")).strip()
            return f"{title}" + (f" — {author}" if author else "")

        with h1:
            st.markdown(
                card_html(
                    "Your stray from the well trodden path",
                    f"{fmt_book(neg)}",
                    f"<span class='badge'>You {float(neg['My Rating']):.0f} vs Avg {float(neg['Average Rating']):.2f} (Δ {float(neg['Delta']):+.2f})</span>",
                    "Wow you really hated this one.",
                    GR_ACCENT,
                ),
                unsafe_allow_html=True,
            )

        with h2:
            st.markdown(
                card_html(
                    "Your diamond in the rough",
                    f"{fmt_book(pos)}",
                    f"<span class='badge'>You {float(pos['My Rating']):.0f} vs Avg {float(pos['Average Rating']):.2f} (Δ {float(pos['Delta']):+.2f})</span>",
                    "You're good at finding gems",
                    GR_ACCENT_2,
                ),
                unsafe_allow_html=True,
            )

        with h3:
            if underdog is None:
                st.markdown(
                    card_html(
                        "Highlight 3",
                        "(not available)",
                        "<span class='badge'>No Average Rating values found</span>",
                        "",
                        "#8A2BE2",  # bright purple accent
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    card_html(
                        "Your underdog book",
                        f"{fmt_book(underdog)}",
                        f"<span class='badge'>Avg {float(underdog['Average Rating']):.2f}</span>",
                        "You like to give underdogs a chance.",
                        "#8A2BE2",
                    ),
                    unsafe_allow_html=True,
                )

st.divider()

# -----------------------------
# Reviews
# -----------------------------
st.subheader("Your reviews (coverage + extremes)")

if books_read == 0:
    st.info("No read books in {YEAR} to analyze.")
else:
    r = read_yr.copy()
    r["Review Length"] = review_len(r["My Review"])
    has_review = r["My Review"].fillna("").astype(str).str.strip().ne("")
    with_review = r[has_review].copy()

    c1, c2 = st.columns(2)
    c1.metric("Books read with a review", f"{len(with_review):,}")
    c2.metric("Review rate", f"{(len(with_review) / max(len(r), 1)) * 100:.1f}%")

    if len(with_review) == 0:
        st.info(f"No reviews found on books read in {YEAR}.")
    else:
        longest = with_review.sort_values("Review Length", ascending=False).iloc[0]
        shortest = with_review.sort_values("Review Length", ascending=True).iloc[0]

        st.markdown("**Longest review**")
        st.write(f"{longest.get('Title','(unknown)')} — {int(longest['Review Length'])} characters")
        st.write(longest.get("My Review", ""))

        st.markdown("**Shortest review**")
        st.write(f"{shortest.get('Title','(unknown)')} — {int(shortest['Review Length'])} characters")
        st.write(shortest.get("My Review", ""))

st.divider()

# -----------------------------
# Drilldowns (fix index issue)
# -----------------------------
# -----------------------------
# Drilldowns (fix index issue + React crash)
# -----------------------------
st.subheader("Drilldowns")

tab1, tab2, tab3 = st.tabs([f"Read in {YEAR}", f"Added in {YEAR}", f"In your {YEAR} universe"])

with tab1:
    if len(read_yr) == 0:
        st.info(f"No books with Date Read in {YEAR}.")
    else:
        show_cols = [c for c in ["Title","Author","Date Read","My Rating","Average Rating","Number of Pages","Original Publication Year","Exclusive Shelf"] if c in read_yr.columns]
        table = read_yr.sort_values("Date Read", ascending=False)[show_cols].reset_index(drop=True)
        table = sanitize_for_streamlit(table)  # <-- add this
        st.dataframe(table, use_container_width=True, hide_index=True)

with tab2:
    if len(added_yr) == 0:
        st.info(f"No books with Date Added in {YEAR}.")
    else:
        show_cols = [c for c in ["Title","Author","Date Added","Exclusive Shelf","My Rating","Average Rating"] if c in added_yr.columns]
        table = added_yr.sort_values("Date Added", ascending=False)[show_cols].reset_index(drop=True)
        table = sanitize_for_streamlit(table)  # <-- add this
        st.dataframe(table, use_container_width=True, hide_index=True)

with tab3:
    if len(in_yr) == 0:
        st.info(f"No books added or read in {YEAR}.")
    else:
        show_cols = [c for c in ["Title","Author","Date Added","Date Read","Exclusive Shelf","My Rating","Average Rating"] if c in in_yr.columns]
        table = in_yr.sort_values(["Date Read","Date Added"], ascending=False)[show_cols].reset_index(drop=True)
        table = sanitize_for_streamlit(table)  # <-- add this
        st.dataframe(table, use_container_width=True, hide_index=True)


st.caption("sorry if this breaks lol")
