import altair as alt

@alt.theme.register("latex", enable=True)
def theme():
    return {
        "config": {
            "title": {
                "font": "serif",
                "fontWeight": "normal",
                "fontSize": 16,
            },
            "axis": {
                "titleFont": "serif",
                "titleFontWeight": "normal",
                "titleFontSize": 16,
                "labelFont": "serif",
                "labelFontWeight": "normal",
                "labelFontSize": 16,
            },
            "headerColumn": {
                "titleFont": "serif",
                "titleFontWeight": "normal",
                "titleFontSize": 16,
                "labelFont": "serif",
                "labelFontWeight": "normal",
                "labelFontSize": 16,
            },
            "headerRow": {
                "titleFont": "serif",
                "titleFontWeight": "normal",
                "titleFontSize": 16,
                "labelFont": "serif",
                "labelFontWeight": "normal",
                "labelFontSize": 16,
            },
            "legend": {
                "titleFont": "serif",
                "titleFontWeight": "normal",
                "titleFontSize": 16,
                "labelFont": "serif",
                "labelFontWeight": "normal",
                "labelFontSize": 16,
            },
            "text": {
                "font": "serif",
                "fontSize": 14,
            },
        },
    }


@alt.theme.register("latex", enable=True)
def theme_small():
    return {
        "config": {
            "title": {
                "font": "serif",
                "fontWeight": "normal",
                "fontSize": 18,
            },
            "axis": {
                "titleFont": "serif",
                "titleFontWeight": "normal",
                "titleFontSize": 18,
                "labelFont": "serif",
                "labelFontWeight": "normal",
                "labelFontSize": 18,
            },
            "headerColumn": {
                "titleFont": "serif",
                "titleFontWeight": "normal",
                "titleFontSize": 18,
                "labelFont": "serif",
                "labelFontWeight": "normal",
                "labelFontSize": 18,
            },
            "headerRow": {
                "titleFont": "serif",
                "titleFontWeight": "normal",
                "titleFontSize": 18,
                "labelFont": "serif",
                "labelFontWeight": "normal",
                "labelFontSize": 18,
            },
            "legend": {
                "titleFont": "serif",
                "titleFontWeight": "normal",
                "titleFontSize": 18,
                "labelFont": "serif",
                "labelFontWeight": "normal",
                "labelFontSize": 18,
            },
            "text": {
                "font": "serif",
                "fontSize": 18,
            },
        },
    }
