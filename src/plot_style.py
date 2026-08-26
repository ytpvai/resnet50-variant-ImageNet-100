from matplotlib import rcParams
from pathlib import Path

def apply_style():
    """设定画图风格"""

    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejuVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,

        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",

        "lines.linewidth": 1.5,
        "lines.markersize": 4,

        # 去除图例外层方框
        "legend.frameon": False,

        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })

PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
]

def add_panel_label(ax, label, x=-0.12, y=1.05):
    """给某个子图的左上角加一个编号标注"""

    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=12
        )

def save(fig: "Figure", save_path: Path):
    """保存图片为 png 和 pdf 格式"""

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = str(save_path).rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")