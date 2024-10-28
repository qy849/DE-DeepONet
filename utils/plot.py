import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_truth_pred_err(X, Y, truth, pred, error):

    fig = plt.figure(figsize=(14, 5))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(1, 3),
                    axes_pad=0.7,
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="each",
                    cbar_size="5%",
                    cbar_pad="2%",
                    )
    
    norm1 = mcolors.Normalize(vmin=min(truth.min(),pred.min()), 
                              vmax=max(truth.max(),pred.max()))
    norm2 = mcolors.LogNorm(vmin=error.min()+numpy.finfo(numpy.float64).eps, # numpy.finfo(numpy.float64).eps = 2.220446049250313e-16
                            vmax=error.max()+numpy.finfo(numpy.float64).eps)

    image1 = grid[0].pcolormesh(X, Y, truth, cmap='turbo', norm=norm1, shading='gouraud')
    image2 = grid[1].pcolormesh(X, Y, pred, cmap='turbo', norm=norm1, shading='gouraud')
    image3 = grid[2].pcolormesh(X, Y, error, cmap='turbo', norm=norm2, shading='gouraud')

    grid.cbar_axes[0].colorbar(image1)
    grid.cbar_axes[1].colorbar(image2)
    grid.cbar_axes[2].colorbar(image3)

    grid[0].set_title('Ground Truth')
    grid[1].set_title('Prediction')
    grid[2].set_title('Absolute Error')
    plt.close()

    return fig