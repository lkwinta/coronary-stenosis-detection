import matplotlib.pyplot as plt


def show_image(*imgs, titles=None, cmap="gray", size=5):
    """Display one or more images side by side.

    Parameters
    ----------
    *imgs : numpy.ndarray
        Images to display (2D grayscale or 3D RGB).
    titles : list of str, optional
        Title for each image.
    cmap : str or list of str, optional
        Colormap for grayscale images. A single string applies to all.
        A list assigns a colormap per image (None = default colormap).
        Default is ``"gray"``.
    size : int, optional
        Size of each panel in inches. Default is ``5``.
    """
    if not imgs:
        raise ValueError("At least one image must be provided.")

    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(size * n, size))
    if n == 1:
        axes = [axes]
    for i, img in enumerate(imgs):
        if isinstance(cmap, list):
            c = cmap[i] if i < len(cmap) else None
        else:
            c = cmap
        if img.ndim == 3:
            axes[i].imshow(img)
        else:
            axes[i].imshow(img, cmap=c)
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_lines(data, *groups, size=5):
    """Plot line charts side by side from a dictionary.

    Parameters
    ----------
    data : dict[str, list]
        Metric names mapped to lists of values.
    *groups : tuple of (list of str, str)
        Each group is ``(keys, title)`` defining one subplot.
    size : int, optional
        Size of each panel in inches. Default is ``5``.
    """
    if not groups:
        raise ValueError("At least one group must be provided.")

    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(size * n, size))
    if n == 1:
        axes = [axes]
    for ax, (keys, title) in zip(axes, groups):
        for key in keys:
            ax.plot(data[key], label=key)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)
