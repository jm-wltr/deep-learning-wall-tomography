import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# 1. Load your DXF
doc = ezdxf.readfile("sections_generator/output/sectionsRot/00001_rot.dxf")
msp = doc.modelspace()

# 2. Set up Matplotlib
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis("off")            # optional: hide axes
ax.set_aspect("equal")    # preserve scale

# 3. Render with ezdxf’s Frontend/Backend
ctx     = RenderContext(doc)
backend = MatplotlibBackend(ax)
frontend = Frontend(ctx, backend)
frontend.draw_layout(msp, finalize=True)

# 4. Show or save
plt.show()
# fig.savefig("preview.png", dpi=300)