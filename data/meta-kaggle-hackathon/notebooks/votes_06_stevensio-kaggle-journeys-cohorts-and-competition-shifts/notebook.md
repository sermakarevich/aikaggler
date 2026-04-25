# Kaggle Journeys: Cohorts and Competition Shifts

- **Author:** Steven Sio
- **Votes:** 28
- **Ref:** stevensio/kaggle-journeys-cohorts-and-competition-shifts
- **URL:** https://www.kaggle.com/code/stevensio/kaggle-journeys-cohorts-and-competition-shifts
- **Last run:** 2025-07-16 09:40:41.070000

---

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; font-family: serif; font-size: 18px; line-height: 1.6;">

<h1 style="color: black;">Introduction</h1>    

<p style="text-align: justify; color: black;">
    This analysis explores how Kaggle competition participation has evolved, both overall and by user cohorts. I first looked at the historical count of final team submissions, breaking it down by competition segment, accelerator usage, team size, and medal outcomes, along with trends in engagement scores. 
</p>

<p style="text-align: justify; color: black;">
    I then applied a retrospective cohort analysis, grouping users by the year they joined their first competition (focusing on the 2019, 2020, 2023, and 2024 cohorts). This helped track shifts in behavior: competition preferences, dropout patterns, performance tiers, accelerator usage, team sizes, and time gaps between competitions. To complement this, I conducted a cross-sectional analysis of 2021–2024 submissions using Empirical Cumulative Distribution Function (ECDF) plots to examine how team size, competition segment, medal outcomes, and accelerator usage impact submission timing.
</p>

<p style="text-align: justify; color: black;">
    <strong style="color: black;">The time series</strong> shows that submission counts have grown, mainly driven by Playground competitions, alongside rising entry-level GPU adoption and a shift toward solo participation. Engagement scores have generally increased, though the adjusted engagement score remains stable and still favors higher-tier users. 
</p>
  
<p style="text-align: justify; color: black;">
    <strong style="color: black;">From the cohort study</strong>, while most users still compete only once, a smaller group returns, improves, and moves up the tiers. However, recent cohorts behave differently from 2019–2020: they're more likely to compete solo, favor Playground competitions, have shorter gaps between competitions, and increasingly use entry GPUs over high-end hardware. The ECDF also shows a higher tendency to submit their final versions earlier.
</p>

<p style="text-align: justify; color: black;">
    <strong style="color: black;">The cross-sectional ECDF analysis</strong> reveals consistent patterns across years: bigger teams, medalists, accelerator users, and Research/Featured participants submit later, while solo teams, non-medalists, Playground participants, and even some high-tier solo users submit earlier.
</p>

<p style="text-align: justify; color: black;">
    Overall, Kaggle's user base is becoming more independent, casual, and faster-paced, possibly driven by more efficient workflows or by the impact of GenAI tools post-2019. Future studies could explore how these emerging patterns, especially among this year’s users, impact competition performance, dropout rates, and segment preferences in their next few competitions.
</p>
  
<p style="color: black;"><strong style="color: black;">Analysis Overview:</strong></p>

<p style="color: black;"><strong style="color: black;">Part 1: Time Series Overview: Submissions & Engagement</strong><br>
- <em style="color: black;">Trends in submission counts and engagement over time.</em></p>

<p style="color: black;"><strong style="color: black;">Part 2: A Cohort Study of The Competitive Journey and Recent Trends</strong><br>
- <em style="color: black;">Tracks submission pacing, gaps between competitions, and shifting behaviors across cohorts.</em></p>

<p style="color: black;"><strong style="color: black;">Part 3: Cross-Sectional Analysis of Final Submission Timing</strong><br>
- <em style="color: black;">ECDF analysis of when teams submit, based on team size, accelerators, medal outcomes, and competition type.</em></p>

</div>

# Importing and Functions

```python
import kagglehub

MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")

print("Path to Meta-Kaggle dataset files:", MK_PATH)
```

## Import Libraries

```python
import pandas as pd
import os
from datetime import datetime
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import duckdb
from plotly.offline import init_notebook_mode 
init_notebook_mode(connected=True) # to fix plotly graphs not showing in viewer mode
# from IPython.display import IFrame 
import plotly.io as pio
pio.renderers.default = 'iframe' # use default iframe renderer if not specified
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas.io.formats.format')
warnings.simplefilter('ignore', category=FutureWarning)
```

## Define Plotting functions

```python
segmentOrder = {'CompSegment': ['Playground', 'Featured', 'Research']}
segmentColors = {
    'Playground': '#1f77b4',  # blue
    'Featured': '#ff7f0e',    # orange
    'Research': '#2ca02c',    # green
    #'Community': '#d62728',   # red
    #'Recruitment': '#9467bd'  # purple
}

medalOrder = {'MedalType': ['None', 'Bronze', 'Silver','Gold']}
medalColors = {
    'Gold':   '#FFD700',  # gold
    'Silver': '#C0C0C0',  # silver
    'Bronze': '#CD7F32',  # bronze
    'None':   '#CCCCFF'   # Periwinkle
}

tierOrder = {'UserPerformanceTierName':['Unranked','Expert','Master','Grandmaster']}  #,'Staff'
tierColors = {
    'Unranked': '#20BEFF',      # Kaggle blue
    'Expert':      '#8751FD',  # Purple
    'Master':      '#F97B48',  # Orange-Red
    'Grandmaster': '#E5D050',  # White Gold
    #'Staff':       '#8B0000'   # dark red
}

accelOrder = {'AccelGroup':['Unknown','None','Entry GPU (K80/T4x2/L4x1)', 'High-End GPU (P100/A100/L4x4)','TPU (v2-32/v3-8/VM v3-8)']}
accelColors = {
    'None': '#CCCCFF',  # periwinkle
    'Unknown':'#cccccc',
    'Entry GPU (K80/T4x2/L4x1)': '#91bfdb',  # light blue
    'High-End GPU (P100/A100/L4x4)': '#fc8d59',  # warm orange
    'TPU (v2-32/v3-8/VM v3-8)': '#d73027'  # strong red
}
teamSizeOrder = {'TeamSizeCat':['Solo','Small (2-3)','Big (4+)']}
teamSizeColors = {
    'Solo': '#52b69a',
    'Small (2-3)': '#1a759f',
    'Big (4+)': '#184e77'
}


def get_user_info_by_Id(user_id: int) -> pd.Series:
    if user_id in users.index:
        return users.loc[user_id]
    else:
        return pd.Series(index=users.columns, dtype=object)


def fig_heatmap_cat_cat(users_data, cat1, cat2, title, xaxis_title=None, yaxis_title=None,  normalize='row',level='UserId'):
    """
    Creates a heatmap visualizing the relationship between two categorical variables 
    (`cat1` vs `cat2`) in the given data, with counts aggregated by either unique users 
    (`UserId`) or total submissions (`ScriptId`).

    Normalization can be applied by row, column, or total to show proportions instead of raw counts.
    Axes are optionally reordered based on predefined category orderings.

    Returns a Plotly heatmap figure.
    """
    df = users_data[[level, cat1, cat2]].dropna()

    if level == 'UserId':
        # Count unique users per cat1-cat2 combo
        count_matrix = df.groupby([cat1, cat2])[level].nunique().unstack(fill_value=0)
    elif level == 'ScriptId':
        # Count submissions (rows) per cat1-cat2 combo
        count_matrix = df.groupby([cat1, cat2]).size().unstack(fill_value=0)
    
    # Define orders
    orders = {
        **segmentOrder,
        **medalOrder,
        **tierOrder,
        **accelOrder,
        **teamSizeOrder
    }

    # Reorder rows and columns if orders are known
    if cat1 in orders:
        count_matrix = count_matrix.reindex(orders[cat1])
    if cat2 in orders:
        count_matrix = count_matrix[orders[cat2]]

    # Normalize
    if normalize == 'row':
        norm_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    elif normalize == 'column':
        norm_matrix = count_matrix.div(count_matrix.sum(axis=0), axis=1)
    elif normalize == 'total':
        norm_matrix = count_matrix / count_matrix.values.sum()
    else:
        norm_matrix = count_matrix  # No normalization

    # Plot
    fig = px.imshow(
        norm_matrix,
        text_auto='.2f',
        labels=dict(x=cat2, y=cat1, color='Proportion'),
        color_continuous_scale='Blues',
        title=title
    )
    # change to default axis title if None
    if xaxis_title == None: xaxis_title = cat2
    if yaxis_title == None: yaxis_title = cat1
    
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template='plotly_white',
        height=600
    )

    return fig


def fig_heatmap_cat_cat_grid(users_data, cat1, cat2, years, normalize='row',level='UserId',title=None):
    """
    Input:
      - users_data (DataFrame)
      - cat1, cat2 (str): categorical columns
      - years (list of int): years to facet by
      - normalize (str): normalization method for counts
      - level (str): 'UserId' or 'ScriptId' aggregation
      - title (str, optional): main title
    Output: Plotly figure with grid of heatmaps by year
    """

    cols = len(years)
    fig = make_subplots(rows=1, 
                        cols=cols, 
                        subplot_titles=[f'Submissions in {y}' for y in years],
                       horizontal_spacing=0.02)
    if title == None: title = f'Heatmaps of {cat1} vs {cat2} by {year_col}'
    # update y axis title
    #fig.update_yaxes(title_text=cat1, row=1, col=1)
    
    for i, year in enumerate(years):
        subset = users_data[users_data['SubmissionDate'].dt.year == year]
        heatmap_fig = fig_heatmap_cat_cat(
            subset, cat1, cat2,
            title='',  # Title is handled by subplot_titles
            normalize=normalize,
            level=level
        )

        # Extract heatmap trace(s) from the generated fig
        for trace in heatmap_fig.data:
            fig.add_trace(trace, row=1, col=i+1)

        # Update x axes title per subplot
        fig.update_xaxes(title=dict(text=cat2,font_size=16), showticklabels=True,tickfont_size=15, row=1, col=i+1)
                # Update y axis title only for first plot
        if i == 0:
            fig.update_yaxes(title=dict(text=cat1,font_size=16), showticklabels=True,tickfont_size=15, row=1, col=i+1)
        else:
            fig.update_yaxes(showticklabels=False, row=1, col=i+1)


    fig.update_layout(
        title_text=title,
        title_font_size=18,
        template='plotly_white',
        #height=600,
        width=500 * cols,  # width per number of cols
        coloraxis=dict(colorscale='Blues')
    )
   

    return fig



def fig_comp_completed_counts(users_data, title, group='CompSegment'):
    """
    Input:
      - users_data (DataFrame)
      - title (str): figure title
      - group (str): categorical grouping column
    Output: Plotly stacked bar chart of competition completion counts with drop-off line
    """   
    # change palettes and order based on group
    if group == 'CompSegment':
        colormap = segmentColors
        categoryOrder = segmentOrder
        legend = 'Competition Segment'
    elif group =='UserPerformanceTierName':
        colormap = tierColors
        categoryOrder = tierOrder
        legend = 'Performance Tier'
    elif group == 'MedalType':
        colormap = medalColors
        categoryOrder = medalOrder
        legend = 'Medal Rank'
    elif group =='AccelGroup':
        colormap = accelColors
        categoryOrder = accelOrder
        legend = 'Accelerator Type'
    elif group =='TeamSizeCat':
        colormap = teamSizeColors
        categoryOrder = teamSizeOrder
        legend = 'Team Size'
    else:
        colormap = None
        categoryOrder = None
        legend = group
    # prepare data
    compCountsBySegment = (
        users_data
        .groupby(['UserCompNumber', group])
        .size()
        .reset_index(name='Count')
    )
    compCounts = (
        users_data
        .groupby(['UserCompNumber'])
        .size()
        .reset_index(name='Count')
    )
    compCounts['DropRate'] = compCounts['Count'].pct_change().fillna(0) * -100
    compCounts['Count'] = compCounts['Count'].astype(int)

    # stacked bar base
    fig_counts = px.bar(
        compCountsBySegment,
        x='UserCompNumber',
        y='Count',
        color=group,
        color_discrete_map=colormap,
        category_orders=categoryOrder,
        labels={
            'UserCompNumber':  'n<sup>th</sup> Competition Completed',
            'Count': 'Number of Participants'
        },
        title=title,
        text='Count'
    )
    fig_counts.update_traces(textposition='inside')
    fig_counts.update_layout(
        barmode='stack',
        legend_title_text=legend,
        xaxis=dict(dtick=1, range=[0, 25.5])
    )

    # layer traces
    fig_counts = go.Figure(fig_counts)

    # add drop-off line
    fig_counts.add_trace(go.Scatter(
        x=compCounts['UserCompNumber'],
        y=compCounts['DropRate'],
        mode='lines+markers',
        name='Drop-Off Rate (%)',
        yaxis='y2',
        line=dict(color='firebrick', width=2, dash='dash')
    ))

    # add totals on top of each bar as a text
    fig_counts.add_trace(go.Scatter(
        x=compCounts['UserCompNumber'],
        y=compCounts['Count'],
        text=compCounts['Count'],
        mode='text',
        textposition='top center',
        textfont=dict(size=10, color='black', style='italic',weight='bold'),
        showlegend=False
    ))

    # final axes
    fig_counts.update_layout(
        width=1000,
        margin=dict(l=80, r=100, t=80, b=80),
        legend=dict(
            title_text=legend,
            x=1.08, xanchor='left', yanchor='top'
        ),
        xaxis=dict(dtick=1, range=[0, 15.5]),
        yaxis=dict(title='Number of Participants',
                  rangemode='tozero'),
        yaxis2=dict(
            title='Drop-Off Rate (%)',
            overlaying='y',
            side='right',
            #range=[0, 100],
            tickmode='sync',
            showgrid=True,
            rangemode='tozero'
        )
    )

    return fig_counts


def fig_comp_completed_props(users_data, title, group='CompSegment'):
    """
    Input:
      - users_data (DataFrame)
      - title (str): figure title
      - group (str): categorical grouping column
    Output: Plotly stacked bar chart of competition completion proportions
    """    
    # change palettes and order based on group
    if group == 'CompSegment':
        colormap = segmentColors
        categoryOrder = segmentOrder
        legend = 'Competition Segment'
    elif group =='UserPerformanceTierName':
        colormap = tierColors
        categoryOrder = tierOrder
        legend = 'Performance Tier'
    elif group == 'MedalType':
        colormap = medalColors
        categoryOrder = medalOrder
        legend = 'Medal Rank'
    elif group =='AccelGroup':
        colormap = accelColors
        categoryOrder = accelOrder
        legend = 'Accelerator Type'
    elif group =='TeamSizeCat':
        colormap = teamSizeColors
        categoryOrder = teamSizeOrder
        legend = 'Team Size'
    else:
        colormap = None
        categoryOrder = None
        legend = group    
        
    # Compute proportions within each CompNumber
    compCountsBySegment = users_data.groupby(['UserCompNumber', group]).size().reset_index(name='Count')
    compCountsBySegment['Proportion'] = (
        compCountsBySegment
        .groupby('UserCompNumber')['Count']
        .transform(lambda x: x / x.sum())
    )
    
    fig_props = px.bar(
        compCountsBySegment,
        x='UserCompNumber',
        y='Proportion',
        color=group,
        color_discrete_map=colormap,
        category_orders=categoryOrder,
        text=compCountsBySegment['Proportion'].apply(lambda x: f'{x:.0%}'),
        labels={
            'UserCompNumber': ' n<sup>th</sup> Competition Completed',
            'Proportion': f'{legend} Share'
        },
        title=title
    )
    
    fig_props.update_traces(textposition='inside')
    fig_props.update_layout(
        barmode='stack',
        legend_title_text=legend,
        xaxis=dict(dtick=1, range=[0, 10.5]),
        yaxis=dict(tickformat='.0%', range=[0, 1])
    )
    
    return fig_props

def fig_comp_completed_combined(fig_counts,fig_props,title,xlim=[0,20.5],legend='Competition Segment'):
    """
    Input:
      - fig_counts: Plotly figure of counts
      - fig_props: Plotly figure of proportions
      - title (str): main figure title
      - xlim (list): x-axis limits
      - legend (str): legend title
    Output: Combined Plotly figure with counts + drop-off line above proportions bar chart
    """
    combined = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.3, 0.2],
        specs=[[{"secondary_y": True}], [{}]]
    )
    
    # trace from fig_counts into row1
    for trace in fig_counts.data:
        # determine if this trace should go on secondary y:
        use_secondary = getattr(trace, "yaxis", "") == "y2"
        combined.add_trace(trace, row=1, col=1, secondary_y=use_secondary)
    
    # trace from fig_props into row2
    for trace in fig_props.data:
        trace.showlegend = False  # Disable duplicate legend
        combined.add_trace(trace, row=2, col=1)
    
    # figure level layout settings
    combined.update_layout(
        #width=1000,
        height=800,
        barmode="stack",
        #legend=fig_counts.layout.legend,
        legend_title_text=legend,
        title_text=title,
        title_x=0.05,               # 0 = left, 0.5 = center, 1 = right
        title_y=0.96,              
        title_font=dict(size=20)  
    )
    
    # main titles from both figs
    combined.add_annotation(
        text=fig_counts.layout.title.text,
        xref="paper", yref="paper",
        x=0, y=1.05,  # top-left, above row 1
        showarrow=False,
        font=dict(size=12, family="Arial", color="black"),
        align="left"
    )
    
    combined.add_annotation(
        text=fig_props.layout.title.text,
        xref="paper", yref="paper",
        x=0, y=0.39, 
        showarrow=False,
        font=dict(size=12, family="Arial", color="black"),
        align="left"
    )
    
    # axes titles and ticks
    combined.update_xaxes(title_text="n<sup>th</sup> Competition Completed", dtick=1, row=2, col=1)
    combined.update_xaxes(showticklabels=True, row=1, col=1)
    combined.update_xaxes(range=xlim)
    combined.update_xaxes(
        dtick=1,                  # tick every 1 unit
        tickmode='linear',        
        row=1, col=1
    )
    
    combined.update_xaxes(
        dtick=1,
        tickmode='linear',
        row=2, col=1
    )
    
    combined.update_yaxes(title_text="Number of Participants", rangemode='tozero', row=1, col=1, secondary_y=False)
    combined.update_yaxes(title_text="Drop-Off Rate (%)", rangemode='tozero', row=1, col=1,tickmode='sync',secondary_y=True)
    combined.update_yaxes(title_text=f"{legend} Share", tickformat=".0%", row=2, col=1)
    
    return combined

def fig_time_since_last_comp(users_data,title,xlim=[1.5, 10.5],ylim=None,legend_text=None,stat='Median'):    
    """
    Input:
      - users_data (DataFrame) with 'UserCompNumber', 'UserDaysSinceLastComp', 'UserPerformanceTierName'
      - title (str): figure title
      - xlim, ylim (list, optional): axis limits
      - stat (str): 'Median' or 'Mean' for aggregation
    Output: Plotly line chart of median/mean days since last competition by competition number and tier
    """
    # filter data
    time_since_last_comp = users_data[users_data['UserIsFirstComp'] == False][[
        'UserCompNumber', 'UserIsMedalFirstComp', 'UserDaysSinceLastComp', 'UserPerformanceTierName'
    ]].dropna(subset=['UserCompNumber', 'UserDaysSinceLastComp'])

    # choose to extract mean or median data based on stat parameter
    if stat=='Median':
        # Group by CompNumber and Performance Tier, compute median
        time_since_last_comp_agg = time_since_last_comp.groupby(
            ['UserCompNumber', 'UserPerformanceTierName']
        )['UserDaysSinceLastComp'].median().reset_index()
        
    elif stat=='Mean':
        # Group by CompNumber and Performance Tier, compute mean
        time_since_last_comp_agg = time_since_last_comp.groupby(
            ['UserCompNumber', 'UserPerformanceTierName']
        )['UserDaysSinceLastComp'].mean().reset_index()
    
    # Line plot
    fig = px.line(
        time_since_last_comp_agg,
        x='UserCompNumber',
        y='UserDaysSinceLastComp',
        color='UserPerformanceTierName',
        color_discrete_map=tierColors,
        category_orders=tierOrder,
        markers=True,  
        labels={'UserDaysSinceLastComp': f"{stat} Days Since Prior Competition",
               'UserCompNumber':'n<sup>th</sup> Competition Submission'},
        title=title
    )
    
    # Set axis limits
    fig.update_layout(
        xaxis=dict(range=xlim,
                  dtick=1),
        yaxis=dict(range=ylim),        
        legend=dict(
            title_text='Performance Tier',
                x=1.08,
                xanchor='left',
                yanchor='top'
            )
    )
    return fig
    

def fig_time_since_last_comp_by_cohort(users_data, years, title=None, xlim=[1.5, 4.5], ylim=None,stat='Median'):
    """
    Input:
      - users_data (DataFrame)
      - years (list): cohort years
      - title (str, optional): figure title
      - xlim, ylim (list, optional): axis limits
      - stat (str): 'Median' or 'Mean'
    Output: Plotly subplot figure with time since last competition by cohort year
    """   
    if title is None:
        title = "Median Days Since Last Competition by Cohorts"

    subplot_titles = [f"<b>First Competition in {year}</b>" for year in years]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        #shared_xaxes=False,
        shared_yaxes=True,
        vertical_spacing=0.16,
        row_heights=[0.2, 0.2],
    )

    positions = {
        years[0]: (1, 1),
        years[1]: (1, 2),
        years[2]: (2, 1),
        years[3]: (2, 2)
    }

    for i, year in enumerate(years):
        row, col = positions[year]
        data_subset = users_data[users_data['UserFirstCompYear'] == year]

        subfig = fig_time_since_last_comp(
            data_subset,
            title='',
            xlim=xlim,
            ylim=ylim,
            legend_text='Performance Tier',
            stat=stat
        )

        for trace in subfig['data']:
            if i > 0:
                trace.showlegend = False  # Only show legend once
            fig.add_trace(trace, row=row, col=col)

        fig.update_xaxes(range=xlim, title_text="n-th Competition", row=row, col=col, dtick=1)

        # Only set y-axis title on left column
        if col == 1:
            fig.update_yaxes(
                title_text=f"{stat} Days",
                range=ylim,
                row=row,
                col=col
            )
        else:
            fig.update_yaxes(range=ylim, row=row, col=col)

    fig.update_layout(
        height=600,
        title_text=title,
        showlegend=True,
        legend=dict(
            title='Performance Tier',
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top'
        )
    )

    fig.update_annotations(font=dict(size=12))

    return fig


def fig_comp_completed_by_cohort(users_data,years,title=None,group='CompSegment',legend='Competition Segment',ylim=None):  
    """
    Input:
      - users_data (DataFrame)
      - years (list): cohort years
      - title (str, optional): main title
      - group (str): grouping column for category
      - legend (str): legend title
      - ylim (list, optional): y-axis limits
    Output: Plotly subplot figure of competition completion counts and proportions by cohort
    """    
    subplot_titles = []
    if title == None:
        title = f"<b>Participation and {legend} Share Based On First Year of Competition<b>"
    for y in years:
        subplot_titles.append(f"<b>First Competition in {y}</b>")
        subplot_titles.append(f"<b>First Competition in {y}</b>")
        
    # Create 12 rows × 2 cols
    fig = make_subplots(
        rows=len(years),
        cols=2,
        specs=[[{"secondary_y": True}, {}] for _ in years], # col 1 has 2 axis, col 2 is normal
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.09,
        horizontal_spacing=0.2,
        subplot_titles=subplot_titles
    )
    
    # loop over years
    for i, year in enumerate(years, start=1): #enumerate() produces (index, value) pairs, index start at 1
        users_data_year = users_data[users_data["UserFirstCompYear"] == year]
        
        # Generate the two figures
        fig_counts = fig_comp_completed_counts(
            users_data_year,
            group=group,
            title=""
        )
        fig_props = fig_comp_completed_props(
            users_data_year,
            group=group,
            title=""
        )
        
        # left column: Counts (1st axis) + Drop-off rate (2nd axis)
        for trace in fig_counts.data:
            is_secondary = getattr(trace, "yaxis", "") == "y2"
            fig.add_trace(trace, row=i, col=1, secondary_y=is_secondary)
        
        # right column: Proportions
        for trace in fig_props.data:
            fig.add_trace(trace, row=i, col=2, secondary_y=False)
    
        # Update axes
        fig.update_xaxes(title_text="n-th Competition", title_font=dict(size=11),tickfont=dict(size=9), dtick=1,range=[0, 10.5], row=i, col=1)
        fig.update_xaxes(title_text="n-th Competition", title_font=dict(size=11),tickfont=dict(size=9), dtick=1, range=[0, 10.5], row=i, col=2)
        fig.update_yaxes(title_text="Participants", title_font=dict(size=11),tickfont=dict(size=9), row=i, col=1, range=ylim, secondary_y=False)
        fig.update_yaxes(title_text="Drop-Off Rate (%)", title_font=dict(size=11),tickfont=dict(size=9), range=[0, 100], row=i, col=1, secondary_y=True, tickmode='sync')
        fig.update_yaxes(title_text=f"{legend} Share", title_font=dict(size=11), tickfont=dict(size=9),tickformat=".0%", row=i, col=2)
    
    # Final layout
    fig.update_layout(
        barmode="stack",
        height=900,
        title={
            "text": title,
            "x": 0.05,            
            "xanchor": "left",  # anchor the left edge of the text at x=0.1
            "font": {
                "size": 20
            }
        },
        #legend_title_text=legend,
        legend=dict(
            font=dict(size=11),
            title=dict(text=f"<b>{legend}</b>")
    )
    )
    
    fig.update_annotations(font=dict(size=12))

    # handle duplicate legends
    seen = set()
    for t in fig.data:
        if t.name in seen:
            t.showlegend = False
        else:
            seen.add(t.name)
    
    return fig

    
def fig_ecdf_overall(users_data,title,group='UserPerformanceTierName'):
    """
    Input:
      - users_data (DataFrame) with columns ['TeamId','CompDaysUsedPct', group]
      - title (str): figure title
      - group (str): categorical grouping column
    Output: Plotly ECDF plot of competition duration used proportion by group
    """
    # change palettes and order based on group
    if group == 'CompSegment':
        colormap = segmentColors
        categoryOrder = segmentOrder
        legend = 'Competition Segment'
    elif group =='UserPerformanceTierName':
        colormap = tierColors
        categoryOrder = tierOrder
        legend = 'Performance Tier'
    elif group == 'MedalType':
        colormap = medalColors
        categoryOrder = medalOrder
        legend = 'Medal Rank'
    elif group =='AccelGroup':
        colormap = accelColors
        categoryOrder = accelOrder
        legend = 'Accelerator Type'
    elif group =='TeamSizeCat':
        colormap = teamSizeColors
        categoryOrder = teamSizeOrder
        legend = 'Team Size'
    elif group == 'UserFirstCompYear':
        colormap = None  
        categoryOrder = {'UserFirstCompYear': [2019, 2020, 2023, 2024]}  # this is a fix for ordering the legend of one of the plot 
        legend = 'Cohort Year'
    else:
        colormap = None
        categoryOrder = None
        legend = group
    # select team-level submissions data only  
    users_data = users_data[['TeamId','CompDaysUsedPct',group]].drop_duplicates()
    fig = px.ecdf(
        data_frame=users_data,
        x="CompDaysUsedPct",
        color=group,
        color_discrete_map=colormap,
        category_orders=categoryOrder,
        markers=True,   
        title=title,
    )
    
    fig.update_layout(
        xaxis_title="Percentage of Competition Duration Used",
        yaxis_title="Cumulative Proportion of Submissions",
        legend=dict(
            font=dict(size=11),
            title=dict(text=f"<b>{legend}</b>")
        )
    )
    return fig



def fig_ecdf_by_submission_year(users_data, years, title, group='UserPerformanceTierName'):
    """
    Input:
      - users_data (DataFrame) with submission date column
      - years (list): list of years for facetting
      - title (str): figure title
      - group (str): grouping variable
    Output: Plotly 2x2 grid of ECDF plots by submission year
    """    
    # change palettes and order based on group
    if group=='UserPerformanceTierName':
        colormap = tierColors
        categoryOrder = tierOrder
        legend = 'Performance Tier'
    elif group == 'CompSegment':
        colormap = segmentColors
        categoryOrder = segmentOrder
        legend = 'Competition Segment'
    elif group == 'MedalType':
        colormap = medalColors
        categoryOrder = medalOrder
        legend = 'Medal Rank'
    elif group =='AccelGroup':
        colormap = accelColors
        categoryOrder = accelOrder
        legend = 'Accelerator Type'
    elif group =='TeamSizeCat':
        colormap = teamSizeColors
        categoryOrder = teamSizeOrder
        legend = 'Team Size'
    else:
        colormap = None
        categoryOrder = None
        legend = group
    # Filter only submissions from selected years
    # select team-level submissions data only  
    data_all_year = (
        users_data.loc[users_data['SubmissionDate'].dt.year.isin(years), 
                       ['TeamId', 'CompDaysUsedPct', 'SubmissionDate',group]].drop_duplicates()
    )
    # Create subplot figure: 2x2 grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"<b>Submissions in {y}</b>" for y in years],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    # Row/column mapping
    positions = {
        years[0]: (1, 1),
        years[1]: (1, 2),
        years[2]: (2, 1),
        years[3]: (2, 2)
    }
    
    # Loop through years and add ECDF plots
    for year in years:
        row, col = positions[year]
        data_year = data_all_year[data_all_year['SubmissionDate'].dt.year == year]
    
        ecdf_fig = px.ecdf(
            data_frame=data_year,
            x="CompDaysUsedPct",
            color=group,
            color_discrete_map=colormap,
            category_orders=categoryOrder,
            markers=True,
        )
    
        for trace in ecdf_fig.data:
            fig.add_trace(trace, row=row, col=col)
    
    # Final layout
    fig.update_layout(
        height=800,
        width=900,
        title_text=title,
        title_x=0.05,
        showlegend=True,
        legend=dict(
            font=dict(size=11),
            title=dict(text=f"<b>{legend}</b>")
        )
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Competition Duration Used (%)")
    fig.update_yaxes(title_text="Cumulative Proportion")
    
    # Optional: Reduce duplicate legends
    seen = set()
    for trace in fig.data:
        if trace.name in seen:
            trace.showlegend = False
        else:
            seen.add(trace.name)
    return fig


def fig_time_series_submissions_ma(users_data, group, title, min_year=2019, window=30, min_periods=1):
    """
    Input:
      - users_data (DataFrame) with 'SubmissionDate', group column, and 'ScriptId'
      - group (str): categorical grouping column
      - title (str): figure title
      - min_year (int): filter submissions from this year onwards
      - window (int): rolling window size (days)
      - min_periods (int): minimum periods for rolling average
    Output: Plotly line chart of rolling mean submission counts over time by group
    """    
    # Pick color map and category order
    if group == 'UserPerformanceTierName':
        colormap = tierColors
        category_order = tierOrder[group]
        legend_title = 'Performance Tier'
    elif group == 'CompSegment':
        colormap = segmentColors
        category_order = segmentOrder[group]
        legend_title = 'Competition Segment'
    elif group == 'MedalType':
        colormap = medalColors
        category_order = medalOrder[group]
        legend_title = 'Medal Rank'
    elif group == 'AccelGroup':
        colormap = accelColors
        category_order = accelOrder[group]
        legend_title = 'Accelerator Type'
    elif group == 'TeamSizeCat':
        colormap = teamSizeColors
        category_order = teamSizeOrder[group]
        legend_title = 'Team Size'
    else:
        colormap = None
        category_order = sorted(users_data[group].dropna().unique())
        legend_title = group

    # Prep time series in long format
    df = users_data[users_data['SubmissionDate'].dt.year >= min_year]
    df = df.groupby(['SubmissionDate', group])['ScriptId'].count().reset_index(name='SubmissionCount')
    
    # compute moving average in wide format
    df_wide = df.pivot(index='SubmissionDate', columns=group, values='SubmissionCount').sort_index()
    df_wide = df_wide.rolling(window=window, min_periods=min_periods).mean()
    
    # pivot back to long format for line plot
    df_long = df_wide.reset_index().melt(id_vars='SubmissionDate', var_name=group, value_name='SubmissionCount_MA')
    fig = px.line(
        df_long, 
        x="SubmissionDate", 
        y="SubmissionCount_MA", 
        color=group,
        color_discrete_map=colormap,
        category_orders={group: category_order},
        title=title,
        labels={
            'SubmissionDate': 'Submission Date',
            'SubmissionCount_MA': f'Submission Count - {window}-Day MA',
            group: legend_title  # label for legend
        }
    )
    
    fig.update_layout(
        legend_title_text=legend_title
    )
    
    return fig


def fig_time_series_submissions_ma_grid(users_data,groups,titles,
                                        main_title,
                                        window=60,
                                        min_periods=1,
                                        min_year=2019,
                                        height=1200):
    """
    Input:
      - users_data (DataFrame)
      - groups (list of str): list of grouping columns for separate plots
      - titles (list of str): subplot titles
      - main_title (str): overall figure title
      - window, min_periods, min_year: parameters for rolling mean calculation
      - height (int): figure height
    Output: Plotly vertical grid subplot figure of rolling mean submission counts by multiple groups
    """
    # Create 3-row subplot
    fig_subplots = make_subplots(
        rows=len(groups), cols=1,
        shared_xaxes=False,
        subplot_titles=titles,
        vertical_spacing=0.1
    )
    
    for i, group in enumerate(groups):
        fig = fig_time_series_submissions_ma(
            users_data,
            group=group,
            title='',
            min_year=min_year,
            window=window,
            min_periods=min_periods
        )
    
        seen = set()  # Track which legend labels already added
        for trace in fig.data:
            show_legend = trace.name not in seen
            seen.add(trace.name)
    
            fig_subplots.add_trace(
                go.Scatter(
                    x=trace.x,
                    y=trace.y,
                    name=trace.name,
                    mode='lines',
                    line=dict(color=trace.line.color),
                    legendgroup=f"{group}",      # keep legend group per subplot
                    showlegend=show_legend       # only show once per label
                ),
                row=i+1, col=1
            )
    
        # Manually update axis titles per row
        fig_subplots.update_yaxes(title_text=f'Submissions {window}-Days MA', row=i+1, col=1)
        fig_subplots.update_xaxes(title_text='Submission Date', row=i+1, col=1)
    
    # calculate gap between legend groups
    fig_subplots.update_layout(
        margin=dict(t=100, b=40)
    )
    
    usable_height = height - fig_subplots.layout.margin.t - fig_subplots.layout.margin.b
    row_height = usable_height / len(groups)
    legend_tracegroupgap = row_height * 0.75
    
    # Final layout 
    fig_subplots.update_layout(
        height=height,
        title_text=main_title,
        template='plotly_white',
        legend_tracegroupgap=legend_tracegroupgap,  
        margin=dict(t=100, b=40)
    )
    
    return fig_subplots


def fig_time_series_mean_ma(users_data, group, title, y='KernelEngagement',yname='Engagement Score', min_year=2019, window=30, min_periods=1):
    """
    Input:
      - users_data (DataFrame) with date, group column, and metric column y
      - group (str): grouping column
      - title (str): figure title
      - y (str): metric column name
      - yname (str): metric display name for labels
      - min_year (int): filter start year
      - window (int): rolling window size
      - min_periods (int): minimum periods for rolling mean
    Output: Plotly line chart of rolling mean metric over time by group
    """
    # Pick color map and category order
    if group == 'UserPerformanceTierName':
        colormap = tierColors
        category_order = tierOrder[group]
        legend_title = 'Performance Tier'
    elif group == 'CompSegment':
        colormap = segmentColors
        category_order = segmentOrder[group]
        legend_title = 'Competition Segment'
    elif group == 'MedalType':
        colormap = medalColors
        category_order = medalOrder[group]
        legend_title = 'Medal Rank'
    elif group == 'AccelGroup':
        colormap = accelColors
        category_order = accelOrder[group]
        legend_title = 'Accelerator Type'
    elif group == 'TeamSizeCat':
        colormap = teamSizeColors
        category_order = teamSizeOrder[group]
        legend_title = 'Team Size'
    else:
        colormap = None
        category_order = sorted(users_data[group].dropna().unique())
        legend_title = group

    # Prep time series in long format
    df = users_data[users_data['SubmissionDate'].dt.year >= min_year]
    df = df.groupby(['SubmissionDate', group])[y].mean().reset_index(name='AvgMetric')
    
    # compute moving average in wide format
    df_wide = df.pivot(index='SubmissionDate', columns=group, values='AvgMetric').sort_index()
    df_wide = df_wide.rolling(window=window, min_periods=min_periods).mean()
    
    # pivot back to long format for line plot
    df_long = df_wide.reset_index().melt(id_vars='SubmissionDate', var_name=group, value_name='AvgMetric_MA')
    fig = px.line(
        df_long, 
        x="SubmissionDate", 
        y="AvgMetric_MA", 
        color=group,
        color_discrete_map=colormap,
        category_orders={group: category_order},
        title=title,
        labels={
            'SubmissionDate': 'Submission Date',
            'AvgMetric_MA': f'Mean {yname} - {window}-Day MA',
            group: legend_title  # label for legend
        }
    )
    
    fig.update_layout(
        legend_title_text=legend_title
    )
    
    return fig
```

# Data Preparation

## Importing data

```python
# getting fields needed:            
submissions = pd.read_csv(f"{MK_PATH}/Submissions.csv",
                          usecols=["Id","TeamId","SourceKernelVersionId","SubmissionDate","IsAfterDeadline"])  #"PrivateScoreLeaderboardDisplay","IsSelected",
submissions['SubmissionDate'] = pd.to_datetime(submissions['SubmissionDate'])


teamMemberships = pd.read_csv(f"{MK_PATH}/TeamMemberships.csv",
                             usecols=["Id","TeamId","UserId"])


competitions = pd.read_csv(f"{MK_PATH}/Competitions.csv",
                          usecols=['Id','HostSegmentTitle','EnabledDate','DeadlineDate','Title'])
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate']).dt.floor('D') # keep only the date component to avoid negative comp days
competitions['DeadlineDate'] = pd.to_datetime(competitions['DeadlineDate']).dt.floor('D') 


kernels = pd.read_csv(f"{MK_PATH}/Kernels.csv",
                     usecols=['Id','Medal','TotalVotes','TotalComments','TotalViews']) #'CurrentUrlSlug',


versions = pd.read_csv(f"{MK_PATH}/KernelVersions.csv",
                      usecols=['Id','ScriptId','AcceleratorTypeId']) #,'RunningTimeInMilliseconds','TotalLines'

accelerators = pd.read_csv(f"{MK_PATH}/KernelAcceleratorTypes.csv")

teams = pd.read_csv(f"{MK_PATH}/Teams.csv",
                   usecols=['Id','CompetitionId','TeamLeaderId','Medal','IsBenchmark'])


users = pd.read_csv(f"{MK_PATH}/Users.csv",
                   usecols=["Id","UserName","PerformanceTier"])
users.set_index("Id", inplace = True)

tier_map = {
    #0: 'Unranked',  # Originally 'Novice', Retired
    1: 'Unranked',  # Originally 'Contributor', Retired
    2: 'Expert',
    3: 'Master',
    4: 'Grandmaster',
    5: 'Staff'  
}
users['PerformanceTierName'] = users['PerformanceTier'].map(tier_map)
```

The following SQL query retrieves team details along with their kernel submissions details. Only the most recent kernel version that were submitted before competition deadline are included, with the submission date reflecting each team's most recent submission.

```python
team_submissions = duckdb.query(
    '''
    SELECT *
    FROM (
        SELECT 
            submissions.TeamId,
            submissions.SourceKernelVersionId,
            submissions.SubmissionDate,
            ROW_NUMBER() OVER (PARTITION BY TeamId ORDER BY SubmissionDate DESC) AS row,
            teams.CompetitionId,
            teams.Medal
        FROM submissions LEFT JOIN teams ON submissions.TeamId = teams.Id
        WHERE 
            submissions.IsAfterDeadline == False AND
            submissions.SourceKernelVersionId IS NOT NULL AND
            teams.IsBenchmark == False
    ) teams
    WHERE row = 1;
    ''').to_df()

team_submissions.head()
```

The next query retrieves the user IDs of all team members from teams that successfully submitted kernels to competitions. I also mapped competition details and final kernel metadata to each submission using data from `KernelVersions.csv`, `Kernels.csv`, and `Competitions.csv`.  

***Note:** This analysis excludes competitions from the `Getting Started`, `Analytics`, `Community` and `Recruitment` segments, focusing only on competitions that are: `Featured`, `Playground`, or `Research`.*

```python
user_team_submissions = duckdb.query(
    '''
    SELECT 
        teamMemberships.TeamId AS TeamId,
        teamMemberships.UserId AS UserId,
        team_submissions.CompetitionId AS CompetitionId,
        team_submissions.Medal AS TeamMedal,
        team_submissions.SubmissionDate AS SubmissionDate,
        versions.ScriptId AS ScriptId,
        accelerators.Label AS Accelerator,
        competitions.HostSegmentTitle AS CompSegment,
        competitions.Title AS CompTitle,
        competitions.EnabledDate AS EnabledDate,
        competitions.DeadlineDate AS DeadlineDate,
        kernels.Medal AS KernelMedal,
        kernels.TotalViews AS KernelViews,
        kernels.TotalVotes AS KernelVotes,
        kernels.TotalComments AS KernelComments
        --versions.RunningTimeInMilliseconds AS KernelRuntime,
        --versions.TotalLines AS KernelTotalLines
    FROM teamMemberships 
    INNER JOIN team_submissions  ON teamMemberships.TeamId = team_submissions.TeamId
    INNER JOIN versions ON team_submissions.SourceKernelVersionId = versions.Id
    INNER JOIN competitions ON team_submissions.CompetitionId = competitions.Id
    LEFT JOIN kernels ON versions.ScriptId = kernels.Id
    LEFT JOIN accelerators ON versions.AcceleratorTypeId = accelerators.Id
    ''').to_df()

# get more columns
user_info = user_team_submissions['UserId'].apply(get_user_info_by_Id)
user_team_submissions = user_team_submissions.join(user_info)

# filter out staff users
user_team_submissions = user_team_submissions[~(user_team_submissions['PerformanceTierName']=='Staff')]

# filter out 'Community','Getting Started', 'Analytics' and 'Recruitment' competitions
user_team_submissions = user_team_submissions[~user_team_submissions['CompSegment'].isin(['Community', 'Recruitment','Analytics','Getting Started'])]

user_team_submissions.head()
```

Note that each row corresponds to a unique `TeamId`–`UserId`–`ScriptId` combination. Although any of these IDs can repeat across rows, no combination of all three appears more than once.

Next, I’ll use this dataframe to figure out how many competitions each user has joined and create some new features.

```python
users_comp_stats = duckdb.query(
    '''
    SELECT 
        UserId,
        UserName,
        TeamId,
        ScriptId,
        Accelerator,
        CompetitionId,
        PerformanceTierName AS UserPerformanceTierName, 
        SubmissionDate,
        CompSegment,
        CompTitle,
        EnabledDate,
        DeadlineDate,
        ROW_NUMBER() OVER (PARTITION BY UserId ORDER BY SubmissionDate ASC) AS UserCompNumber,
        TeamMedal AS Medal,
        KernelViews,
        KernelVotes,
        KernelComments
        --KernelRuntime,
        --KernelTotalLines
    FROM user_team_submissions
    ''').to_df()
users_comp_stats['Medal'] = users_comp_stats['Medal'].fillna(0)
users_comp_stats['Accelerator'] = users_comp_stats['Accelerator'].fillna('None')
#users_comp_stats['KernelMedal'] = users_comp_stats['KernelMedal'].fillna(0)

# map medal type
medal_mapping = {0: 'None', 1: 'Gold', 2: 'Silver', 3: 'Bronze'}
users_comp_stats['MedalType'] = users_comp_stats['Medal'].map(medal_mapping)

# map accelerators into 4 main types
accel_group_map = {
    'None': 'None',
    # Entry GPUs
    'GPU K80': 'Entry GPU (K80/T4x2/L4x1)',
    'GPU T4 x2': 'Entry GPU (K80/T4x2/L4x1)',
    'GPU L4 x1': 'Entry GPU (K80/T4x2/L4x1)',
    # High-End GPUs
    'GPU P100': 'High-End GPU (P100/A100/L4x4)',
    'GPU A100': 'High-End GPU (P100/A100/L4x4)',
    'GPU L4 x4': 'High-End GPU (P100/A100/L4x4)',
    # TPUs
    'TPU v2-32': 'TPU (v2-32/v3-8/VM v3-8)',
    'TPU v3-8': 'TPU (v2-32/v3-8/VM v3-8)',
    'TPU VM v3-8': 'TPU (v2-32/v3-8/VM v3-8)'
}
users_comp_stats['AccelGroup'] = users_comp_stats['Accelerator'].map(accel_group_map).fillna('Unknown')

# calculate days since last competition
#users_comp_stats['SubmissionDate'] = pd.to_datetime(users_comp_stats['SubmissionDate'])
users_comp_stats = users_comp_stats.sort_values(by=['UserId', 'SubmissionDate'])
users_comp_stats['UserDaysSinceLastComp'] = users_comp_stats.groupby('UserId')['SubmissionDate'].diff().dt.days
users_comp_stats['UserDaysSinceLastComp'] = users_comp_stats['UserDaysSinceLastComp'].fillna(0)

# compute days used for competition
users_comp_stats['CompDaysUsed'] = (users_comp_stats['SubmissionDate'] - users_comp_stats['EnabledDate']).dt.days

# compute days used as percentage of whole comp duration
users_comp_stats['CompDaysUsedPct'] = users_comp_stats['CompDaysUsed'] / (users_comp_stats['DeadlineDate'] - users_comp_stats['EnabledDate']).dt.days 
users_comp_stats['CompDaysUsedPct'] = users_comp_stats['CompDaysUsedPct'].clip(upper=1) # cap at 1 and 0
users_comp_stats['CompDaysUsedPct'] = users_comp_stats['CompDaysUsedPct'].clip(lower=0)

# compute IsSolo indicator
team_sizes = users_comp_stats.groupby('TeamId')['UserId'].nunique().reset_index(name='TeamSize')
users_comp_stats = users_comp_stats.merge(team_sizes, on='TeamId')
users_comp_stats['IsSolo'] = users_comp_stats['TeamSize'] == 1

# team size categories
users_comp_stats['TeamSizeCat'] = pd.cut(
    users_comp_stats['TeamSize'],
    bins=[0,1,3,float('inf')],
    labels=['Solo','Small (2-3)','Big (4+)'],
    right=True,
    include_lowest=True
)

# compute IsFirstComp indicator
users_comp_stats['UserIsFirstComp'] = users_comp_stats['UserCompNumber'] == 1

# compute IsMedal indicator
users_comp_stats['IsMedal'] = users_comp_stats['Medal'] > 0

# compute the year of first competition completion
first_comp_dates = users_comp_stats[users_comp_stats['UserCompNumber']==1][['UserId','SubmissionDate']]
first_comp_dates['UserFirstCompYear'] = first_comp_dates['SubmissionDate'].dt.year
users_comp_stats = users_comp_stats.merge(first_comp_dates[['UserId', 'UserFirstCompYear']], 
                          on=['UserId'],
                          how='left')

# exclude submission records < 2015 
users_comp_stats = users_comp_stats[users_comp_stats['SubmissionDate'].dt.year >= 2015]

# compute proxy engagement score
users_comp_stats['KernelEngagement'] = (users_comp_stats['KernelVotes'] + users_comp_stats['KernelComments']) / (users_comp_stats['KernelViews'] + 1)

# compute adjusted version of engagement (log-denominator)
users_comp_stats['KernelAdjustedEngagement'] = (users_comp_stats['KernelVotes'] + users_comp_stats['KernelComments']) / np.log(users_comp_stats['KernelViews'] + 1)

# drop any duplicates
users_comp_stats = users_comp_stats.drop_duplicates()

users_comp_stats.head()
```

```python
unique_users = len(users_comp_stats['UserId'].unique())
na_users = len(users_comp_stats[(users_comp_stats['UserPerformanceTierName'].isna())]['UserId'].unique())
negative_days_used = len(users_comp_stats[users_comp_stats['CompDaysUsed']<0]['UserId'].unique())
print(f"Among the {unique_users} unique users who have joined at least one competition:\n")
print(f"{na_users} of them have missing values in 'PerformanceTier'")
print(f"{negative_days_used} of them have negative days used for competition\n")

print("I will now remove rows with or negative day values.")
# clean nulls and negative days
users_comp_stats = users_comp_stats.dropna()
users_comp_stats = users_comp_stats[~(users_comp_stats['CompDaysUsed']<0)]

print(f"There are {len(users_comp_stats['TeamId'].unique())} unique teams and {len(users_comp_stats['UserId'].unique())} unique users after cleaning")
print(f"\nMinimum number of competitions joined: {users_comp_stats['UserCompNumber'].min()}")
print(f"\nMaximum number of competitions joined: {users_comp_stats['UserCompNumber'].max()}")
print(f"\nTotal Medals combined for all users:\n{users_comp_stats['Medal'].value_counts()}")
print(f"\nTotal Competition participants in different segments:\n{users_comp_stats['CompSegment'].value_counts()}")
print(f"\nCount of Accelerator usage by competition participants:\n{users_comp_stats['AccelGroup'].value_counts()}")
```

I will now compute some more features after filtering out unclean data.

```python
# create indicator columns for each medal type
users_comp_stats['GoldMedal'] = (users_comp_stats['Medal'] == 1).astype(int)
users_comp_stats['SilverMedal'] = (users_comp_stats['Medal'] == 2).astype(int)
users_comp_stats['BronzeMedal'] = (users_comp_stats['Medal'] == 3).astype(int)
users_comp_stats['IsMedalInt'] = (users_comp_stats['IsMedal']).astype(int)

# compute total medals and comps currently:
user_medal_totals = (
    users_comp_stats.groupby('UserId')
    .agg(
        UserTotalMedalsFinal = ('IsMedalInt', 'sum'),
        UserGoldsFinal = ('GoldMedal', 'sum'),
        UserSilversFinal = ('SilverMedal', 'sum'),
        UserBronzesFinal = ('BronzeMedal', 'sum'),
        UserTotalComps = ('UserCompNumber', 'max')
    )
    .reset_index()
)
users_comp_stats = users_comp_stats.merge(
    user_medal_totals,
    on='UserId',
    how='left'
)
'''
# compute cumulative gold, silver, bronze medals for the user as of submission date:
users_comp_stats = users_comp_stats.sort_values(['UserId', 'SubmissionDate'])
users_comp_stats['UserCumulativeGolds'] = users_comp_stats.groupby('UserId')['GoldMedal'].cumsum()
users_comp_stats['UserCumulativeSilvers'] = users_comp_stats.groupby('UserId')['SilverMedal'].cumsum()
users_comp_stats['UserCumulativeBronzes'] = users_comp_stats.groupby('UserId')['BronzeMedal'].cumsum()
users_comp_stats['UserCumulativeTotalMedals'] = users_comp_stats.groupby('UserId')['IsMedalInt'].cumsum()
'''
# compute MedalNumber (n-th medal of user)
user_medals = duckdb.query(
    '''
        SELECT 
            UserId, 
            UserCompNumber,
            IsMedal,
            ROW_NUMBER() OVER (PARTITION BY UserId ORDER BY UserCompNumber) AS UserMedalNumber
        FROM users_comp_stats
        WHERE IsMedal = TRUE
    ''').to_df()
users_comp_stats = users_comp_stats.merge(user_medals[['UserId', 'UserCompNumber','UserMedalNumber']], 
                          on=['UserId', 'UserCompNumber'],
                          how='left')
users_comp_stats['UserMedalNumber'] = users_comp_stats['UserMedalNumber'].fillna(0)

# compute IsFirstMedal indicator
users_comp_stats['UserIsFirstMedal'] = users_comp_stats['UserMedalNumber'] == 1

# compute IsMedalFirstComp indicator (getting a medal on the first competition)
medal_first_comp_ids = (users_comp_stats[users_comp_stats['UserIsFirstComp'] & users_comp_stats['UserIsFirstMedal']]['UserId'].unique())
users_comp_stats['UserIsMedalFirstComp'] = users_comp_stats['UserId'].isin(medal_first_comp_ids)

# compute submission year
users_comp_stats['SubmissionYear'] = users_comp_stats['SubmissionDate'].dt.year
```

## Exploring Data Peculiarities

```python
# Show users with Medal but still marked Novice
suspicious = users_comp_stats[
    (users_comp_stats['UserPerformanceTierName']=='Unranked') &
    (users_comp_stats['IsMedal'] == True)
]
sus_users = (suspicious[['UserId','UserName','CompetitionId','MedalType','UserPerformanceTierName','CompTitle','UserTotalMedalsFinal']].drop_duplicates())
sus_users.head()
```

```python
# Show users with Medal from Playground Competitions
suspicious2 = users_comp_stats[
    (users_comp_stats['CompSegment']=='Playground') &
    (users_comp_stats['IsMedal'] == True)
]

sus2_users = (suspicious2[['UserId','UserName','CompetitionId','MedalType','UserPerformanceTierName','CompTitle','UserTotalMedalsFinal']].drop_duplicates())
sus2_users.head(5)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 

<p style="color: black;">
I noticed many records of users who have earned at least one medal but still remain in the <strong style="color: black;">Novice</strong> tier (as of June 2025 Kaggle data). At first glance, this seems inconsistent. However, according to <a href="https://www.kaggle.com/progression" style="color: black;">the official Kaggle tier progression logic</a>, these users may have missed some required engagement or profile completion steps, which causes them to remain in the Novice tier (or possibly some other reasons).
</p>

<ul style="color: black;">
<li><em style="color: black;">For example, when I look up the user with username <code>mrboor</code> (ID <code>5411</code>) on Kaggle, he does have 1 Gold medal from a team competition, but remains Unranked for some reason.</em></li>
</ul>

<p style="color: black;">
Secondly, I also noticed there were medals given to Playground competitions in this data, but since <a href="https://www.kaggle.com/progression" style="color: black;">Playground competition Medals typically do not count towards Kaggle ranking Points or Medals</a>. These medals are likely rare cases where they are counted towards competition Medals.  
</p>

<ul style="color: black;">
<li><em style="color: black;">For example, the user <code>superant</code> (ID <code>72177</code>) is shown to be awarded a gold medal from <a href="https://www.kaggle.com/competitions/hungry-geese" style="color: black;">this Playground Competition</a> which does award points and medals for some reason.</em></li>
</ul>

<p style="color: black;">
To avoid misleading results, I will still consider these medals awarded from Playground Competitions since it was shown to be awarded officially as well.
</p>

</div>

```python
# check final dataframe
users_comp_stats.sort_values(by=['UserId','UserCompNumber']).head(5)
```

```python
users_comp_stats.describe(include='all')
```

```python
print(users_comp_stats.info())
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 

<h1 style="text-align: justify;color: black">Final processed data</h1>

<p style="text-align: justify;color: black"> The final processed dataset is structured at the user–competition–kernel level, where each row represents a unique combination of TeamId, UserId, and KernelId. For each team, the final kernel is selected as the submission with the most recent submission date. Although TeamId, UserId, and KernelId can each occur multiple times across the dataset, no combination of all three ever repeats.</p>  
  
<p style="text-align: justify;color: black"> For example, if a kernel was submitted as a team submission, the same KernelId can appear in multiple rows, one for each user on the team. Likewise, a UserId can occur in different rows if the user participated in multiple competitions or teams. In short, each row uniquely identifies one user linked to a specific final kernel within a particular team in a competition.</p>
  
<p style="text-align: justify;color: black"> To reduce inaccuracies from outdated records, I have excluded all competition kernel submissions dated before 2017. Even so, I recognize that the data is still not completely reliable. In some cases, medals awarded to users are missing or not counted correctly based on the given data when cross-checked against official user profiles on the Kaggle website. As a result, this analysis may be partially distorted and could be improved in the future.</p>
</div>

```python
users_comp_stats.info()
```

```python
users_comp_stats.sort_values(by=['UserId','UserCompNumber']).head(5)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;">

<h1 style="color: black;">Part 1: Time Series Overview: Submissions & Engagement</h1>

<p style="color: black;">In this section, I will provide an overview of competition submissions and their kernel engagement over time. Submission counts reflect only the final version of each kernel submitted before the competition deadline. I also break down submission activity by user cohorts, where each cohort consists of users who completed their first competition in the same year.</p>

<p style="color: black;">Engagement is measured by the sum of upvotes and comments per user view. An adjusted engagement score is also included, where the number of views is log-scaled to dampen the effect of inflated view counts. This adjustment helps highlight kernels that receive relatively few upvotes or comments despite high visibility.</p>

<h2 style="color: black;">Time Series Chart of Final Submissions By Attributes</h2>

<p style="color: black;">The time series chart <strong style="color: black;">(Figure 1)</strong> shows a huge boom in final submission counts in 2023 and remains at a steady level ever since. This growth is driven mainly by Playground competitions, with Featured competitions showing a more recent upward trend. In contrast, Research competitions stayed flat, aside from a dip in 2020, likely tied to COVID-19 disruptions.</p>

<p style="color: black;">Entry-level GPU usage also increased starting in late 2022, with gradual growth since. Around the same time, there was a clear rise in solo team participation, pointing to a broader shift toward more individual-driven competition formats.</p>

<p style="color: black;"><em style="color: black;">Note that the numbers shown represent the 60-day moving average instead of the raw count on a single day; this helps smooth out short-term fluctuations for clearer trend visibility.</em></p>

</div>

```python
groups = ['CompSegment', 'TeamSizeCat', 'AccelGroup','MedalType']
titles = [
    'By Competition Segment',
    'By Team Size',
    'By Accelerator usage',
    'By Medal Awarded'
]

fig = fig_time_series_submissions_ma_grid(
    users_comp_stats,
    groups=groups,
    titles=titles,
    min_year=2019,
    main_title='<b>Figure 1: Daily Final Submission Counts Over Time by Attributes</b><br><sup>60-Day Moving Average Line Chart</sup>',
    window=60,
    min_periods=1,
    height=900)
fig.write_html('figure1.html')

fig.show(renderer='iframe')
#IFrame('figure1.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;">

<h2 style="color: black;">Competition Participation Cycle of Users by Cohorts</h2>

<p style="color: black;">When analyzing final submissions by user cohorts - grouped by their first competition year <strong style="color: black;">(Figure 2)</strong>, a clear lifecycle pattern could be seen. Most cohorts tend to ramp up participation, peak mid-year, and then gradually decline over time.</p>

<p style="color: black;">From 2022 to 2024, the peak moving average submission counts increased steadily, reaching a high of roughly 14 in January 2024. However, for the 2025 cohort, the current peak sits around 9 and has been declining since May, suggesting a possible slowdown in participation momentum this year.</p>

</div>

```python
fig = fig_time_series_submissions_ma(users_comp_stats[(users_comp_stats['UserFirstCompYear']>=2019)&(users_comp_stats['IsSolo']==True)],
                   group='UserFirstCompYear',
                   title='<b>Figure 2: Daily Final Submission Counts by Cohorts</b><br><sup>60-Day Moving Average Line Chart. Data includes Kernel Submissions of the <b>Final Version and from Solo Teams only.</sup>',
                   min_year=2019,
                   min_periods=1,
                   window=60)
fig.update_layout(height=500)
fig.write_html('figure2.html')

fig.show(renderer='iframe')
#IFrame('figure2.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 

<h2 style="color: black;">Engagement Levels Over Time by Performance Tier</h2>

<p style="color: black;">Now let’s look at engagement scores of submissions based on user tiers. To prevent skill bias from teams, the chart below <strong style="color: black;">(Figure 3)</strong> includes only kernel submissions from solo teams. This ensures each submission is attributed to a single user and their performance tier, making comparisons more meaningful.</p>

<p style="color: black;">Kernel engagement scores have shown a strong upward trend since 2020, with submissions from Experts recently achieving the highest moving average. However, when using the adjusted engagement score <strong style="color: black;">(Figure 4)</strong>, a clearer pattern emerges: engagement levels follow the Kaggle performance tier hierarchy, where Grandmasters consistently lead (10 or higher), followed by Masters (5–10), then Experts (below 5).</p>

<p style="color: black;"><em>Note:</em> All references to performance tier in this analysis reflect each user’s current highest overall tier.</p>

</div>

```python
fig = fig_time_series_mean_ma(users_comp_stats[users_comp_stats['IsSolo']==True],
                   group='UserPerformanceTierName', y='KernelEngagement', yname='Engagement Score',
                   title='<b>Figure 3: Daily Mean Engagement Score by Current User Tiers</b><br><sup>60-Day Moving Average Line Chart. Data includes Kernel Submissions of the <b>Final Version and from Solo Teams only.</b><br><b>Engagement Score </b>= (Upvotes + Comments)/(Views + 1)</sup>',
                   min_year=2018,
                   min_periods=1,
                   window=60)
fig.update_layout(height=500)
fig.write_html('figure3.html')

fig.show(renderer='iframe')
#IFrame('figure3.html', width='100%',height=600)
```

```python
fig = fig_time_series_mean_ma(users_comp_stats[users_comp_stats['IsSolo']==True],
                   group='UserPerformanceTierName',y='KernelAdjustedEngagement',yname='Adjusted Engagement',
                   title='<b>Figure 4: Daily Mean Adjusted Engagement Score by Current User Tiers</b><br><sup>60-Day Moving Average Line Chart. Data includes Kernel Submissions of the <b>Final Version and from Solo Teams only.</b><br><b>Adjusted Engagement Score</b> = (Upvotes + Comments)/log(Views + 1)</sup>',
                   min_year=2019,
                   min_periods=1,
                   window=60)
fig.update_layout(height=500)
fig.write_html('figure4.html')
fig.show(renderer='iframe')
#IFrame('figure4.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;">

<h2 style="color: black;">Engagement Levels Over Time by Competition Segments</h2>

<p style="color: black;"><strong style="color: black;">Figure 5</strong> shows that submission counts have been climbing steadily since 2020, with the Playground segment leading the growth. Featured and Research competitions have had similar engagement levels over time, but a clear gap has opened up between them and Playground in recent years.</p>

<p style="color: black;">However, when we look at the adjusted engagement score in <strong style="color: black;">Figure 6</strong>, Research and Featured kernels consistently score higher than Playground. This suggests that while Playground competitions get a lot more views, they don’t necessarily generate more meaningful engagement such as upvotes and comments.</p>

</div>

```python
fig = fig_time_series_mean_ma(users_comp_stats,
                   group='CompSegment',y='KernelEngagement',yname='Engagement Score',
                   title='<b>Figure 5: Daily Mean Engagement Score by Competition Segments</b><br><sup>60-Day Moving Average Line Chart. Data includes Kernel Submissions of the Final Version Only.<br><b>Engagement Score </b>= (Upvotes + Comments)/(Views + 1)</sup>',
                   min_year=2019,
                   min_periods=1,
                   window=60)
fig.update_layout(height=500)
fig.write_html('figure5.html')

fig.show(renderer='iframe')
#IFrame('figure5.html', width='100%',height=600)
```

```python
fig = fig_time_series_mean_ma(users_comp_stats,
                   group='CompSegment',y='KernelAdjustedEngagement',yname='Adjusted Engagement',
                   title='<b>Figure 6: Daily Mean Adjusted Engagement Score by Competition Segments</b><br><sup>60-Day Moving Average Line Chart. Data includes Kernel Submissions of the Final Version Only.<br><b>Adjusted Engagement Score </b>= (Upvotes + Comments)/log(Views + 1)</sup>',
                   min_year=2019,
                   min_periods=1,
                   window=60)
fig.update_layout(height=500)
fig.write_html('figure6.html')

fig.show(renderer='iframe')
#IFrame('figure6.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;">

<h1 style="color: black;">Part 2: A Cohort Study of The Competitive Journey and Recent Trends</h1>

<p style="color: black;">In this part, I implemented a retrospective cohort approach to analyze user behavior based on the year of their first competition entry, focusing on the 2019, 2020, 2023, and 2024 cohorts. This approach gives us a clearer view of how user engagement evolves, instead of just looking at the community as one big pool.</p>

<h2 style="color: black;">Overview of How Participation Changes as Users Gain Experience</h2>

<p style="color: black;"><strong style="color: black;">Figure 7</strong> shows how many users completed the n<sup>th</sup> competition in 2015 or later, broken down by competition segments (only includes Playground, Featured, and Research).</p>

<p style="color: black;">Data clearly shows that most users stop competing on Kaggle after their first competition. Although the rate of drop-off is notably high during the first several competitions, it gradually declines and begins to stabilize beyond the seventh competition. The bottom plot shows the proportion of competition segments joined on the n-th competition; the proportions are mostly consistent overall with a greater preference for Featured competitions.</p>

<p style="color: black;"><strong style="color: black;">Note:</strong></p>
<ul style="color: black;">
  <li><em style="color: black;">“Competitions Completed”</em> refers to the number of competitions in which a Kaggle user submitted a kernel before the competition’s submission deadline.</li>
  <li><em style="color: black;">"Number of participants"</em> does not represent the number of teams, it represents all unique individuals within all teams participating in any of the competitions from the segments specified.</li>
  <li style="color: black;">The whole analysis does not include users who had never participated in any competitions.</li>
</ul>

</div>

```python
fig_counts = fig_comp_completed_counts(users_comp_stats,
                                       title="How Many Kagglers Have Completed Their n-th Competition?<br><sup>Most People Stop at the First One.</sup>",
                                      group='CompSegment')
fig_props = fig_comp_completed_props(users_comp_stats,
                                     title="Which Competition Segments do Kagglers join in their n<sup>th</sup> Competition?<br><sup>Segment Share is the percentage of competitions from each segment across all users’ n<sup>th</sup> competition entries.</sup>",
                                    group='CompSegment')

fig = fig_comp_completed_combined(fig_counts,fig_props,title="<b>Figure 7: Competition Participation drop-off and Segment shares</b><br><sup>Users who started competing in 2015 or later, by Competition Segment Joined.</sup>",xlim=[0, 20.5],legend='Competition Segment')
fig.update_layout(height=700)
fig.write_html('figure7.html')
fig.show(renderer='iframe')
#IFrame('figure7.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;">

<p style="color: black;">Secondly, by looking at Kagglers grouped by their current highest performance tier overall (<strong style="color: black;">Figure 8</strong>), we see that the proportion of high-tier participants increases with the number of competitions completed. In other words, <strong style="color: black;">today’s</strong> Experts, Masters, and Grandmasters tend to have completed more competitions than those in lower tiers. The next normalized histogram shows this clearly (<strong style="color: black;">Figure 9</strong>): up to 80% of current unranked Kagglers have competed in only one competition, while more than 40% of Experts, Masters, and Grandmasters have entered multiple competitions.</p>

<p style="color: black;"><strong style="color: black;">Note:</strong></p>
<ul style="color: black;">
  <li style="color: black;">The performance tier shown in this analysis represents each user’s highest overall tier and is not limited to Competition tier alone. As a result, some Experts or even Grandmasters with only one competition may have earned their title through other tracks such as Notebooks, Datasets, or Code.</li>
</ul>

</div>

```python
fig_counts = fig_comp_completed_counts(users_comp_stats,
                                       title="Count of Current Performance Tier of Kagglers on their n-th Competition<br><sup>Most of the Current Novices Stop at the First One.</sup>",
                                      group='UserPerformanceTierName')
fig_props = fig_comp_completed_props(users_comp_stats,
                                     title="Which Performance Tiers Compete More Often?<br><sup>Share of Competitions by Tier at Each n<sup>th</sup> Entry</sup>",
                                    group='UserPerformanceTierName')

fig = fig_comp_completed_combined(fig_counts,fig_props,
                            title="<b>Figure 8: Participation Patterns by Current Performance Tier</b><br><sup>Users who started competing in 2015 or later, grouped by Present Tier (as of 2025)</sup>",
                            xlim=[0, 20.5],
                           legend='Performance Tier')
fig.update_layout(height=700)
fig.write_html('figure8.html')
fig.show(renderer='iframe')
#IFrame('figure8.html', width='100%',height=600)
```

```python
users_total_comp = users_comp_stats[['UserId','UserPerformanceTierName','UserTotalComps']].drop_duplicates()
x0 = users_total_comp[users_total_comp['UserPerformanceTierName']=='Grandmaster']['UserTotalComps']
x1 = users_total_comp[users_total_comp['UserPerformanceTierName']=='Master']['UserTotalComps']
x2 = users_total_comp[users_total_comp['UserPerformanceTierName']=='Expert']['UserTotalComps']
x3 = users_total_comp[users_total_comp['UserPerformanceTierName']=='Unranked']['UserTotalComps']


fig = go.Figure()
fig.add_trace(go.Histogram(
    x=x0,
    histnorm='percent',
    name='Grandmaster', # name used in legend and hover labels
    xbins=dict( # bins used for histogram
        start=0.5,
        end=7.5,
        size=1
    ),
    marker_color=tierColors['Grandmaster'],
    #opacity=0.75
))
fig.add_trace(go.Histogram(
    x=x1,
    histnorm='percent',
    name='Master',
    xbins=dict(
        start=0.5,
        end=7.5,
        size=1
    ),
    marker_color=tierColors['Master'],
    #opacity=0.75
))
fig.add_trace(go.Histogram(
    x=x2,
    histnorm='percent',
    name='Expert',
    xbins=dict(
        start=0.5,
        end=7.5,
        size=1
    ),
    marker_color=tierColors['Expert'],
    #opacity=0.75
))

fig.add_trace(go.Histogram(
    x=x3,
    histnorm='percent',
    name='Unranked',
    xbins=dict(
        start=0.5,
        end=7.5,
        size=1
    ),
    marker_color=tierColors['Unranked'],
    #opacity=0.75
))

fig.update_layout(
    title_text='<b>Figure 9: Normalized Histogram of Total Competitions Completed by Tiers<br></b><sup>For Users With First Competition Starting 2015 or Later</sup>', # title of plot
    xaxis_title_text='Total Competitions', # xaxis label
    xaxis_range=[0,7.5],
    yaxis_title_text='Percent', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)
fig.write_html('figure9.html')

fig.show(renderer='iframe')
#IFrame('figure9.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 
  
<h2 style="color: black;">Competition Participation and Tier Progression by Cohorts</h2>
<p style="color: black;">Let’s examine how users from different <strong style="color: black;">cohorts (defined by the year they entered their first competition)</strong> have progressed in terms of participation and current performance tier <strong style="color: black;">(Figure 10)</strong>. The distribution of tiers across competition counts mirrors earlier trends: users who compete more tend to reach higher tiers, and those who started earlier are more likely to have become Grandmasters.</p>
  
<p style="color: black;"><strong style="color: black;">Figure 11</strong> further supports this by showing a Normalized Count Heatmap of user cohorts by current performance tier. Notably, the majority of current Masters and Grandmasters began competing as early as 2015, with another peak for Masters in 2021 and for Grandmasters in 2020. In contrast, the majority of current Experts and Unranked entered in 2024. This trend highlights that reaching higher tiers like Grandmaster typically requires long-term engagement, whereas lower tiers such as Expert can be achieved more quickly, though still often over multiple years.</p>

</div>

```python
fig = fig_comp_completed_by_cohort(users_comp_stats,
                       title="<b>Figure 10: Participation and Performance Tier Share Based On Cohorts",
                       years=[2019,2020,2023,2024],group='UserPerformanceTierName',legend='Performance Tier', ylim=[0,4000])
fig.update_layout(height=800)
fig.write_html('figure10.html')

fig.show(renderer='iframe') 
#IFrame('figure10.html', width='100%',height=600)
```

```python
fig = fig_heatmap_cat_cat(
    users_comp_stats, 
    cat1='UserFirstCompYear', 
    cat2='UserPerformanceTierName',
    xaxis_title='Performance Tier',
    yaxis_title='First Competition Year (Cohort)',
    normalize='column',
    level='UserId',
    title="<b>Figure 11: Distribution of First Competition Year By Performance Tier</b><br><sup>Count Heatmap of Users in Each Cohort vs Performance Tier, <b>normalized by columns.</sup>"
)
fig.update_yaxes(dtick=1)  
fig.update_layout(height=500)
fig.write_html('figure11.html')

fig.show(renderer='iframe')
#IFrame('figure11.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 
   
<h2 style="color: black;">Shifts in Competition Preferences: From Research to Playground</h2>

<p style="color: black;"><strong style="color: black;">Figure 12</strong> below shows that <strong style="color: black;">the 2019 and 2021 cohort group</strong> initially favored Featured and Research competitions. Over time, many who started in 2019 gradually shifted toward more advanced research problems, while those who started in 2020 often began with Research or Featured competitions but then moved to easier Playground competitions in subsequent entries.</p>

<p style="color: black;"><strong style="color: black;">For the 2023 and 2024 cohort</strong>, with the rise of LLMs and AI-assisted coding, far fewer newcomers started with Research competitions at all. Instead, around 50% of newcomers chose Playground competitions for their first, reflecting a influx of less-experienced users drawn in by the support of AI tools or access to more high-quality online courses. Although 2024 saw roughly 23% more new participants, these users had higher dropout rates after each competition when compared to users starting in 2023, suggesting many were quickly overwhelmed and failed to stay engaged beyond their first few competitions.</p>

<p style="color: black;">Based on the earlier chart of performance tier distribution, it’s clear that high-tier users are the ones who stay active across many competitions. However, this chart shows that once dominated by research-oriented challengers, Kagglers have increasingly tilted toward Playground competitions as the default training ground. This also suggests that many top-tier users increasingly rely on alternative paths to reach their current overall performance tier, such as contributing datasets and notebooks, since Playground competitions generally don’t award medals or points and therefore don’t contribute directly to tier progression. The rapid emergence of new ML techniques may also support this trend, as Playground contests offer a flexible platform for broad experimentation rather than deep specialization in competitions such as Research or Featured.</p>

</div>

```python
fig = fig_comp_completed_by_cohort(users_comp_stats,
                       title="<b>Figure 12: Participation Patterns Based On Cohorts",
                       years=[2019,2020,2023,2024],group='CompSegment',ylim=[0,4000])
fig.update_layout(height=800)
fig.write_html('figure12.html')

fig.show(renderer='iframe')  
#IFrame('figure12.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 

<h2 style="color: black;">Trends in Team Collaboration</h2>
  
<p style="color: black;"><strong style="color: black;">Figure 13</strong> below shows that across all cohorts, users are more likely to join teams in their first competition but increasingly compete solo as they gain experience. However, there's a noticeable shift when comparing the 2019-2020 cohorts to those from 2023-2024: in 2019-2020, at least 15% of users still joined small teams in their second competition, whereas in 2023-2024, that number drops below 10% and even lower for subsequent competitions. This suggests that newer users are transitioning to solo competition earlier than before.</p>  

<p style="color: black;">The reason for this shift isn't entirely clear, but some patterns stand out. The normalized count heatmap <strong style="color: black;">(Figure 14)</strong> shows that for final submissions in 2019, 2020, 2023, and 2024, the majority of medal-winning kernel submissions came from non-solo teams, a trend that has been consistent since 2017.</p>

<p style="color: black;">Additionally, larger teams consistently submit later in competition timelines, suggesting they utilize more of the available time, and also due to greater coordination overhead. This is explored further in a later section.</p>  

<p style="color: black;">Overall, these trends suggest a need to either address the barriers to team collaboration or better support the emerging preference for solo participation.</p>
</div>

```python
fig = fig_comp_completed_by_cohort(users_comp_stats,
                       title="<b>Figure 13: Participation and Team Size Share Based On Cohorts",
                       years=[2019,2020,2023,2024],group='TeamSizeCat',legend='Team Size', ylim=[0,4000])
fig.update_layout(height=800)
fig.write_html('figure13.html')

fig.show(renderer='iframe')  
#IFrame('figure13.html', width='100%',height=600)
```

```python
fig = fig_heatmap_cat_cat_grid(
    users_comp_stats,
    cat1='IsSolo',
    cat2='IsMedal',
    years=[2019,2020,2023,2024],
    normalize='column',
    level='ScriptId', # kernel level count
    title='<b>Figure 14: Count Heatmap of Solo Indicator vs Medal Indicator</b><br><sup>Normalized by Columns.</sup>'
)
fig.update_layout(height=350,
                 width=1000)
fig.write_html('figure14.html')

fig.show(renderer='iframe')
#IFrame('figure14.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 

<h2 style="color: black;">Accelerator Usage Trend</h2>

<p style="color: black;"><strong style="color: black;">Figure 15</strong> below shows which accelerators were used in the cohorts' n-th competition. Competitors who began their <strong style="color: black;">first competition in 2019 or 2020</strong> relied more on high-end accelerators like P100s and A100s during their first 10 competitions, indicating a preference for compute-heavy tasks. In contrast, those starting in <strong style="color: black;">2023 or 2024</strong> show a noticeable decline in accelerator usage both relatively and absolutely, with more relying on no accelerator at all. However, usage of entry-level GPUs is growing within the 2024 cohort.</p>
  
<p style="color: black;">This shift may reflect a move toward more efficient modeling practices that reduce the need for powerful hardware, a lack of technical expertise with high-end accelerators among newer users, or simply a preference for tasks requiring less computational power.</p>

</div>

```python
fig = fig_comp_completed_by_cohort(users_comp_stats,
                       title="<b>Figure 15: Participation and Accelerator Usage Based On Cohorts",
                       years=[2019,2020,2023,2024],group='AccelGroup',legend='Accelerator Used', ylim=[0,4000])
fig.update_layout(height=800)
fig.write_html('figure15.html')

fig.show(renderer='iframe') 
#IFrame('figure15.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 

<h2 style="color: black;">Trends in Time Usage Between Competitions</h2>
<h3 style="color: black;">Kaggle’s Top Users Spend More Time In-Between Competitions</h3>

<p style="color: black;"><strong style="color: black;">Figures 16.1 and 16.2</strong> show the median and mean time gaps (in days) between competitions for overall competitors, grouped by their current performance tier as of 2025. The data reveals that competitors in higher tiers tend to take longer breaks between competitions, especially after their first one.</p>

<p style="color: black;">These time gaps are generally right-skewed, as the mean is consistently higher than the median. For all tiers except Unranked, the gaps generally shrink with more competition experience, stabilizing at around 30–50 median days or 100–150 mean days between submissions over the next few competitions.</p>

</div>

```python
fig = fig_time_since_last_comp(users_comp_stats,
                         title="<b>Figure 16.1: Median Days Since Last Competition's Submission (Overall)</b><br><sup>For All Users Starting in 2015 or Later by Performance Tier Currently.</sup>",
                        xlim=[1.5,7.5],ylim=[0,300])
fig.update_layout(height=450)
fig.write_html('figure16_1.html')

fig.show(renderer='iframe') 
#IFrame('figure16_1.html', width='100%',height=600)
```

```python
fig = fig_time_since_last_comp(users_comp_stats,
                         title="<b>Figure 16.2: Mean Days Since Last Competition's Submission (Overall)</b><br><sup>For All Users Starting in 2015 or Later by Performance Tier Currently.</sup>",
                         stat='Mean',
                        xlim=[1.5,7.5],ylim=[0,300])
fig.update_layout(height=450)
fig.write_html('figure16_2.html')

fig.show(renderer='iframe') 
#IFrame('figure16_2.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 

<p style="color: black;"><strong style="color: black;">Figures 17.1 and 17.2</strong> break these competitors down by cohort. Competitors from the 2019–2020 cohorts show much larger and clearer time gaps between tier groups observed both in terms of median and mean days. In contrast, the 2023–2024 cohorts have shorter and more compressed time gaps across all tiers.</p>

<p style="color: black;">This suggests competitors starting from 2023 have either become much more efficient in participating and completing competitions, or that the rate of new competitions has increased compared to previous years, leading to more frequent participation.</p>

</div>

```python
fig = fig_time_since_last_comp_by_cohort(users_comp_stats,
                                   years=[2019,2020,2023,2024],
                                   ylim=[0,500],
                                   title='<b>Figure 17.1: Median Days Since Last Competition by Cohorts and Tiers')
fig.update_layout(height=600)
fig.write_html('figure17_1.html')

fig.show(renderer='iframe')
#IFrame('figure17_1.html', width='100%',height=600)
```

```python
fig = fig_time_since_last_comp_by_cohort(users_comp_stats,years=[2019,2020,2023,2024],
                                   ylim=[0,500],
                                   stat='Mean',
                                   title='<b>Figure 17.2: Mean Days Since Last Competition by Cohorts and Tiers')
fig.update_layout(height=600)
fig.write_html('figure17_2.html')

fig.show(renderer='iframe')
#IFrame('figure17_2.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 

<h1 style="color: black;">Part 3: Distribution Analysis: Time to Final Submission</h1>

<p style="text-align: justify; color: black;">To better understand how much time each <strong style="color: black;">team</strong> (Solo/Non-Solo) spends on competitions submitted in recent years, I created an Empirical Cumulative Distribution Function (ECDF) plot to visualize the cumulative proportion of final kernel submissions over the course of each competition (measured as a percentage of competition duration). <strong style="color: black;">Figure 18</strong>, for example, shows the overall time spent by teams based on their final medal awarded. This can be interpreted as the probability of completing a competition before a given point in time, conditional on a specific factor (e.g., team size, segment, or hardware used).</p>

<p style="text-align: justify; color: black;">Because competitions vary in length, I standardized the time variable as the percentage of days used relative to the total competition duration.</p>

<strong style="color: black;">Definitions:</strong>
<ul style="color: black;">
  <li><em>Competition Duration</em>: Measured as days from competition enabled date to competition deadline date.</li>
  <li><em>Days used to complete the competition</em>: Measured as days from competition enabled date to last submission date.</li>
  <li><em>Percentage of Competition Duration Used</em>: Computed as the percentage of days taken to complete the competition relative to Competition Duration: Days taken to complete the competition / Competition Duration.</li>
</ul>

</div>

```python
fig = fig_ecdf_overall(users_comp_stats,
                 group='MedalType',
                 title="<b>Figure 18: Time to Final Submission by Medal Awarded (Overall)</b><br><sup>ECDF Plot of Final Submission Timing From Submissions Data (2015 or Later).</sup>")
fig.update_layout(height=500)
fig.write_html('figure18.html')

fig.show(renderer='iframe')
#IFrame('figure18.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 

<p style="color: black;"><strong>To interpret the ECDF plot:</strong> the y-axis represents the cumulative proportion of a final submission occurring at or before the percentage of competition duration shown on the x-axis.</p>

<p style="text-align: justify; color: black;">At the right end of the x-axis (the competition deadline), you’ll notice that the cumulative proportion for each tier jumps vertically. This happens because all remaining submissions made exactly at the deadline are included there. The height of this final jump reflects the proportion of submissions made by teams who waited until the last moment to submit. For example, roughly 50% of all teams awarded Gold made their final submission just before the competition deadline (around 99% of the total duration). The remaining 50% submitted exactly at 100%, right at the deadline date.</p>

<p style="text-align: justify; color: black;">By looking at the curvature of the plot, it shows that teams with no medals have a near-linear curve, meaning there is a consistent rate of final submissions made throughout the competition timeline, and they usually submitted before reaching the end of the deadline. In contrast, the slowest teams were the more successful ones, with a higher rate of final submissions made near the end of the competition.</p>  

<h2 style="color: black;">Time to Submission by Cohorts</h2>
<p style="text-align: justify; color: black;">When comparing time to submission across cohorts based on their first three competitions, the ECDF plot <strong>(Figure 19)</strong> shows that the curves for more recent cohorts (2023–2024) are slightly steeper compared to those from 2019 and 2020. This suggests that newer cohorts are submitting a bit earlier, or at least are less inclined to wait until the very last moment compared to earlier cohorts.</p>

</div>

```python
fig = fig_ecdf_overall(users_comp_stats[(users_comp_stats['UserFirstCompYear'].isin([2019,2020,2023,2024])) &
                       (users_comp_stats['UserCompNumber']<=3)],
                 group='UserFirstCompYear',
                 title="<b>Figure 19: Time to Final Submission By Cohorts (First Three Competitions)</b><br><sup>ECDF Plot of Final Submission Timing From Submissions Data (2015 or Later).</sup>")
fig.update_layout(height=500)
fig.write_html('figure19.html')

fig.show(renderer='iframe')
#IFrame('figure19.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;">

<h2 style="color: black;">Cross-Sectional Distribution Analysis by Submission Year</h2>

<p style="color: black;">The following figures break down the distribution of final submissions by the <strong style="color: black;">year of submission (2021 to 2024), rather than by user cohort</strong>. By analyzing each year separately, we can see how factors like competition segment, team size, and accelerator usage influence when teams tend to submit.</p>

<p style="color: black;">This helps reveal whether certain behaviors or strategies have become more or less common over time.</p>

<h3 style="color: black;">Competition Segment On Time to Submission:</h3>

<p style="color: black;">Submission timing patterns have remained fairly consistent across the past four years <strong style="color: black;">(Figure 20)</strong>. Playground competitions show the steepest and most consistently rising ECDF curves, indicating faster and steadier submission rates.</p>

<p style="color: black;">In contrast, Featured and Research competitions have slightly flatter curves early on, with varying degrees of mid-competition submission speed year to year, likely reflecting differences in difficulty or the level of commitment required each year.</p>

</div>

```python
fig = fig_ecdf_by_submission_year(users_comp_stats, 
                       years=[2021,2022,2023,2024], 
                       title="<b>Figure 20: Time to Final Submission by Competition Segment</b><br><sup>ECDF Plots for Team Submissions Made in 2021–2024, Faceted by Submission Year.</sup>",
                       group='CompSegment')
fig.update_layout(height=650)
fig.write_html('figure20.html')

fig.show(renderer='iframe')
#IFrame('figure20.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 

<h2 style="color: black;">Current Performance Tier On Time to Submission:</h2>

<p style="text-align: justify; color: black;">To avoid bias from mixed-tier teams when analyzing submission timing by performance tier, this part includes only submissions from solo teams. By looking at the curvature of the functions conditional on different tier groups <strong style="color: black;">(Figure 21)</strong>, some distinct submission behaviour is observed:</p>  

<ul style="color: black;">
  <li><strong style="color: black;">Current Masters and Grandmasters</strong> have a much steeper curve in the beginning of their competition, and slow down gradually mid-way or near the end of the competition. This pattern is consistent even in submissions from three years ago and suggests many submit well before the deadline, likely reflecting experience, good preparation, and a strong head start.</li>

  <li><strong style="color: black;">Experts</strong> show a steady, almost linear increase throughout the competition duration, implying a consistent rate of final submissions. Their curve still ends nearly at the same point, but their pacing is more uniform compared to higher tiers.</li>

  <li><strong style="color: black;">Unranked Kagglers</strong> have the slowest curve in the beginning and accelerate mid-way to catch up on other tier groups, likely due to inexperience or a lack of early preparation to submit as quickly as other tier groups.</li>
</ul>

</div>

```python
fig = fig_ecdf_by_submission_year(users_comp_stats[users_comp_stats['IsSolo']==True], 
                       years=[2021,2022,2023,2024], 
                       title="<b>Figure 21: Time to Final Submission by Performance Tier</b><br><sup>ECDF Plots for Team Submissions Made in 2021–2024, Faceted by Submission Year. <b>(This Plot Include Solo Teams Only)</b></sup>",
                       group='UserPerformanceTierName')
fig.update_layout(height=650)
fig.write_html('figure21.html')

fig.show(renderer='iframe')
#IFrame('figure21.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 

<h2 style="color: black;">Team Size On Time to Submission:</h2>  

<p style="text-align: justify; color: black;">Submission timing by team size <strong style="color: black;">(Figure 22)</strong> follows a consistent pattern across most submission years. Around 80% of solo competitors tend to submit before the last day of the competition, compared to roughly 60% for small teams (2–3 members) and 50% for larger teams (4+ members). However, trends for large teams in all years remain slightly uncertain due to limited data.</p> 

<p style="text-align: justify; color: black;">The overall pattern suggests that bigger teams often utilize more of the available competition time than solo teams, likely due to greater coordination overhead and barriers in forming teams in general. This may help explain why solo teams have always been the majority and continue to grow, as working in larger teams can often slow down progress and lead to more last-minute submissions.</p>

</div>

```python
fig = fig_ecdf_by_submission_year(users_comp_stats, 
                       years=[2021,2022,2023,2024], 
                       title="<b>Figure 22: Time to Final Submission by Team Size</b><br><sup>ECDF Plots for Team Submissions Made in 2021–2024, Faceted by Submission Year.</sup>",
                       group='TeamSizeCat')
fig.update_layout(height=650)
fig.write_html('figure22.html')

fig.show(renderer='iframe')
#IFrame('figure22.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 
  
<h2 style="color: black;">Accelerator Usage on Time to Submission:</h2>

<p style="text-align: justify; color: black;">Submission timing patterns vary quite a lot by accelerator type and year as seen in <strong style="color: black;">Figure 23</strong>. The ECDF curves for TPU users from 2022 to 2024 are jagged due to limited and inconsistent usage. In contrast, entry-level GPU usage became more widespread starting in 2022, reflected by smoother, more stable curves year by year. Notably, 2024 and 2022 TPU users appeared to submit earlier than the same type of users in the previous year in general, suggesting a higher time-efficiency for TPU users in those years, though data for those groups are still quite limited.</p>

<p style="text-align: justify; color: black;">Teams that did not use any accelerators generally submitted earlier than those using entry-level or high-end GPUs, which is expected since tasks requiring accelerators tend to be more compute-heavy. Interestingly, TPU users submitted ahead of all other groups in 2022 and 2024 but were the slowest in 2023.</p>

</div>

```python
fig = fig_ecdf_by_submission_year(users_comp_stats, 
                       years=[2021,2022,2023,2024], 
                       title="<b>Figure 23: Time to Final Submission by Accelerators Usage</b><br><sup>ECDF Plots for Team Submissions Made in 2021–2024, Faceted by Submission Year.</sup>",
                       group='AccelGroup')
fig.update_layout(height=650)
fig.write_html('figure23.html')

fig.show(renderer='iframe')
#IFrame('figure23.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;"> 
    
<h2 style="color: black;">Medal Outcome on Time to Submission:</h2>

<p style="text-align: justify; color: black;"><strong style="color: black;">Figure 24</strong> shows an interesting pattern among medalists that clearly sets them apart: most medal-winning participants tend to use nearly the entire duration of the competition before making their final submission, regardless of medal type or year.</p>

<p style="text-align: justify; color: black;">In contrast, non-medalists typically submit much earlier, often well before the competition deadline. This suggests that taking more time correlates with better outcomes, possibly due to extended periods spent on model tuning or strategy planning.</p>

</div>

```python
fig = fig_ecdf_by_submission_year(users_comp_stats, 
                       years=[2021,2022,2023,2024], 
                       title="<b>Figure 24: Time to Final Submission by Medal Awarded</b><br><sup>ECDF Plots for Team Submissions Made in 2021–2024, Faceted by Submission Year.</sup>",
                       group='MedalType')
fig.update_layout(height=650)
fig.write_html('figure24.html')

fig.show(renderer='iframe')
#IFrame('figure24.html', width='100%',height=600)
```

<div style="background-color: #edede9; padding: 15px; border-radius: 5px; color: black; text-align: justify; font-family: serif; font-size: 15px; line-height: 1.6;">

<h1 style="color: black;">Conclusion</h1>
  
<p style="color: black;">This analysis highlights a clear evolution in how Kaggle users approach competitions. Participation has grown overall, but with distinct shifts in behavior: more users are competing solo, favoring Playground competitions, adopting entry-level GPUs, and submitting earlier. Meanwhile, medalists, teams competing in challenging competitions, and larger teams still consistently prefer to optimize their time and resources by submitting closer to deadlines.</p> 

<p style="color: black;">These changes point to a rising trend of faster, more casual competition dynamics, potentially influenced by GenAI tools and maturing workflows. Understanding these patterns is crucial for Kaggle to adapt its platform, whether to encourage more sustained engagement, support team collaborations, or design competitions that align with these new participation styles.</p> 
    
<h1 style="color: black;">Recommendations</h1>

<p style="color: black;">To adapt to the growing trend of solo participation, Kaggle could consider making team collaboration more attractive, especially for newcomers, by introducing features like team matchmaking, mentorship programs, or bonus incentives for team-based submissions. For the growing segment of casual and fast-paced competitors, shorter, time-bound challenges or mini-competitions could better align with their preferences and engagement style. Additionally, Kaggle should actively monitor the influence of GenAI tools, as they appear to be accelerating participation cycles and changing how users approach competitions.</p>

<p style="color: black;">To sustain long-term engagement, Kaggle could implement programs that reward continuous participation and progression, helping prevent dropouts after the first competition. Since Playground competitions are the most popular entry point, Kaggle might also develop clearer pathways or progression frameworks that encourage users to transition into more advanced Research or Featured competitions, keeping them challenged and engaged.</p> 

<p style="color: black;">Future studies could explore how these recent patterns, especially among this year’s users, impact competition performance, dropout rates, and segment preferences in their next few competitions.</p>

</div>