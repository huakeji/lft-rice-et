"""Shared style + map helpers for the AFM manuscript figures.

Imported by every figXX_*.py script in this directory. Reads only the basemap
shapefiles in ../basemap; never touches data.
"""
import os, shutil, tempfile, warnings, logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(HERE, ".."))
BASEMAP = os.path.join(FIGDIR, "basemap")
R = os.path.join(FIGDIR, "..", "reports", "awm_repro_20261015")
M = os.path.join(R, "writing_materials")

MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM

# ---------- fixed colours ----------
MODEL_COL = {"LFT+GRU": "#000000", "LFT": "#D55E00", "GRU": "#0072B2",
             "PFT": "#009E73", "MLP": "#CC79A7", "CBM": "#E69F00"}
MODEL_ORDER = ["LFT+GRU", "LFT", "GRU", "PFT", "MLP", "CBM"]
MODEL_MK = {"LFT+GRU": "o", "LFT": "s", "GRU": "^", "PFT": "D", "MLP": "v", "CBM": "P"}
ZONES = ["R1", "R2", "R3", "R4", "R5", "R6"]
ZONE_COL = dict(zip(ZONES, ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628"]))
ZONE_NAME = {"R1": "South China", "R2": "Middle–lower Yangtze", "R3": "Southwest",
             "R4": "North China", "R5": "Northeast", "R6": "Northwest"}
EC_SITES = [("Jurong", 31.807, 119.217), ("Changling", 44.597, 123.470), ("Panjin", 40.941, 121.960)]
ET0 = r"ET$_\mathrm{o}$"; ETC = r"ET$_\mathrm{c}$"; ETA = r"ET$_\mathrm{a}$"; UNIT = r"mm d$^{-1}$"


def setup():
    for d in ("/Library/Fonts", "/System/Library/Fonts/Supplemental"):
        for f in ("Times New Roman.ttf", "Times New Roman Bold.ttf",
                  "Times New Roman Italic.ttf", "Times New Roman Bold Italic.ttf"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 7.5,
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
        "lines.linewidth": 1.0, "lines.markersize": 3.2, "patch.linewidth": 0.5,
        "savefig.dpi": 600, "figure.dpi": 150, "axes.unicode_minus": True,
    })
    # turn missing-font warnings into errors so DejaVu fallback cannot pass silently
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("error", message=".*findfont.*")
    warnings.filterwarnings("error", message=".*Glyph.*missing.*")


def panel(ax, s, x=-0.02, y=1.02, ha="right", va="bottom", **kw):
    return ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                   ha=ha, va=va, **kw)


def halo(lw=1.6, c="white"):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=lw, foreground=c)]


def save(fig, name):
    for ext in ("pdf", "png", "tif"):
        kw = dict(dpi=600, bbox_inches="tight", pad_inches=0.02)
        if ext == "tif":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), **kw)
    # verify fonts actually embedded
    used = set()
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text():
            used.add(font_manager.FontProperties(family=t.get_fontfamily()).get_family()[0])
    print("saved", name, "fonts requested:", used,
          "resolved:", os.path.basename(font_manager.findfont("Times New Roman", fallback_to_default=False)))


# ---------- maps ----------
def albers():
    import cartopy.crs as ccrs
    return ccrs.AlbersEqualArea(central_longitude=105, central_latitude=0, standard_parallels=(25, 47))


_cache = {}


def provinces():
    import geopandas as gpd
    if "prov" not in _cache:
        g = gpd.read_file(os.path.join(BASEMAP, "中国省界.shp"))
        _cache["prov"] = g.to_crs(albers().proj4_init)
    return _cache["prov"]


def ninedash():
    """九段线.prj has a GBK-encoded name that pyogrio cannot parse; read geometry from
    a temporary copy without .prj and assign the CRS stated in that .prj
    (WGS84 Albers, lon0=105, lat1=25, lat2=47)."""
    import geopandas as gpd
    if "nd" not in _cache:
        d = tempfile.mkdtemp()
        for e in ("shp", "shx", "dbf"):
            shutil.copy(os.path.join(BASEMAP, f"九段线.{e}"), os.path.join(d, f"nd.{e}"))
        g = gpd.read_file(os.path.join(d, "nd.shp"))
        shutil.rmtree(d, ignore_errors=True)
        g = g.set_crs("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +x_0=0 +y_0=0 +datum=WGS84 +units=m")
        _cache["nd"] = g.to_crs(albers().proj4_init)
    return _cache["nd"]


MAIN_EXTENT = [-2.68e6, 2.62e6, 1.72e6, 6.02e6]  # Albers metres (x0,x1,y0,y1)
SCS_EXTENT = [0.25e6, 1.85e6, 0.28e6, 2.55e6]


def basemap(ax, lw=0.35, fill="#F7F7F7", scs=False):
    import cartopy.crs as ccrs
    pr = provinces()
    ax.add_geometries(pr.geometry, crs=albers(), facecolor=fill, edgecolor="#6E6E6E", linewidth=lw, zorder=0.5)
    ax.add_geometries(ninedash().geometry, crs=albers(), facecolor="none", edgecolor="#4D4D4D",
                      linewidth=lw * 1.6 if not scs else lw * 1.8, zorder=0.6)
    ax.set_extent(SCS_EXTENT if scs else MAIN_EXTENT, crs=albers())
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def gridlines(ax, size=6.5, left=True, bottom=True):
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False,
                      linewidth=0.3, color="#9A9A9A", linestyle=(0, (2, 2)), alpha=0.8,
                      xlocs=mticker.FixedLocator(range(70, 141, 10)),
                      ylocs=mticker.FixedLocator(range(10, 61, 10)), rotate_labels=False)
    gl.top_labels = False; gl.right_labels = False
    gl.left_labels = left; gl.bottom_labels = bottom
    gl.xlabel_style = {"size": size}; gl.ylabel_style = {"size": size}
    gl.xpadding = 2; gl.ypadding = 2
    return gl


def scs_inset(fig, ax, w=0.19, h=0.30, pad=0.0):
    """South China Sea inset at lower-right corner of map axes `ax`."""
    import cartopy.crs as ccrs
    pos = ax.get_position()
    ia = fig.add_axes([pos.x1 - w * pos.width - pad, pos.y0 + pad, w * pos.width, h * pos.height],
                      projection=albers())
    basemap(ia, lw=0.3, scs=True)
    ia.set_facecolor("white")
    for s in ia.spines.values():
        s.set_linewidth(0.5); s.set_edgecolor("k")
    return ia


def proj_xy(lon, lat):
    import cartopy.crs as ccrs
    pts = albers().transform_points(ccrs.PlateCarree(), np.asarray(lon, float), np.asarray(lat, float))
    return pts[:, 0], pts[:, 1]


def lat_to_y(lat, lon=105.0):
    lat = np.atleast_1d(np.asarray(lat, float))
    return proj_xy(np.full_like(lat, lon), lat)[1]


def discrete_cmap(name, n, lo=0.0, hi=1.0):
    from matplotlib.colors import ListedColormap
    base = plt.get_cmap(name)
    return ListedColormap([base(v) for v in np.linspace(lo, hi, n)])
