import matplotlib.pyplot as plt


def plot(img1, img2, title1, title2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=20)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(title2, fontsize=20)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()