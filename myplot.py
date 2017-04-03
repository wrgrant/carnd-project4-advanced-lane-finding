import matplotlib.pyplot as plt



def close_event():
    plt.close()


def timed_plot_double(img1, img2, title1='', title2=''):
    # Setup a timer
    # fig = plt.figure()
    # timer = fig.canvas.new_timer(interval=500)
    # timer.add_callback(close_event)

    # Do the plotting
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    f.tight_layout()
    ax1.imshow(img1, cmap='gray')
    ax1.set_title(title1, fontsize=20)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(title2, fontsize=20)

    # Show with timer
    # timer.start()
    #plt.draw()
    plt.show(block=False)
    plt.pause(.00001)
    plt.close()



def plot_double(img1, img2, title1='', title2=''):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    f.tight_layout()
    ax1.imshow(img1, cmap='gray')
    ax1.set_title(title1, fontsize=20)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(title2, fontsize=20)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def plot(img, title=''):
    plt.imshow(img)
    plt.title(title)
    plt.show()
