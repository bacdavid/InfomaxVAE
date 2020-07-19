import numpy as np
import matplotlib.pyplot as plt


def plot_reconstruction(x, vae, save_dir='./results'):
    # Predict
    x_reconstructed = vae.predict(x=x)

    # Plot
    n = x.shape[0]
    size = vae.input_shape
    figure = np.zeros((size[0] * n, size[1], size[-1]))
    for i in range(n):
        x_sample = x_reconstructed[i]
        img = x_sample * 255. if size[-1] > 1 else x_sample
        figure[i * size[0]: (i + 1) * size[1], :, :] = img

    plt.figure(dpi=100, figsize=(10, 100))
    plt.imshow(figure.astype('int')) if size[-1] > 1 else plt.imshow(np.squeeze(figure), cmap='Greys_r')
    plt.savefig(save_dir + '/reconstructed.eps')
    plt.close()


def plot_zsemireconstructed(x, vae, save_dir='./results'):
    # Predict
    z, _, _, c = vae.predict(type='encoder', x=x)

    # Compute
    c_mean = np.mean(c, axis=0)

    # Plot
    n = z.shape[0]
    size = vae.input_shape
    figure = np.zeros((size[0] * n, size[1], size[-1]))
    for i in range(n):
        x_sample = vae.predict(type='decoder', z=[z[i]], c=[c_mean])
        img = x_sample * 255. if size[-1] > 1 else x_sample
        figure[i * size[0]: (i + 1) * size[1], :, :] = img

    plt.figure(dpi=100, figsize=(10, 100))
    plt.imshow(figure.astype('int')) if size[-1] > 1 else plt.imshow(np.squeeze(figure), cmap='Greys_r')
    plt.savefig(save_dir + '/zsemireconstructed.eps')
    plt.close()


def plot_csemireconstructed(x, vae, save_dir='./results'):
    # Predict
    z, _, _, c = vae.predict(type='encoder', x=x)

    # Compute
    z_mean = np.mean(z, axis=0)

    # Plot
    n = c.shape[0]
    size = vae.input_shape
    figure = np.zeros((size[0] * n, size[1], size[-1]))
    for i in range(n):
        x_sample = vae.predict(type='decoder', z=[z_mean], c=[c[i]])
        img = x_sample * 255. if size[-1] > 1 else x_sample
        figure[i * size[0]: (i + 1) * size[1], :, :] = img

    plt.figure(dpi=100, figsize=(10, 100))
    plt.imshow(figure.astype('int')) if size[-1] > 1 else plt.imshow(np.squeeze(figure), cmap='Greys_r')
    plt.savefig(save_dir + '/csemireconstructed.eps')
    plt.close()


def plot_zvariation(x, vae, save_dir='./results'):
    # Predict
    z, _, _, c = vae.predict(type='encoder', x=x)

    # Compute
    c_mean = np.mean(c, axis=0)
    z_mean = np.mean(z, axis=0)
    z_max = np.max(z, axis=0)
    z_min = np.min(z, axis=0)

    # Plot
    n = x.shape[0]
    size = vae.input_shape
    for j in range(z_mean.size):
        grid_z = np.linspace(z_min[j], z_max[j], n)[::-1]
        figure = np.zeros((size[0] * n, size[1], size[-1]))
        for i, yi in enumerate(grid_z):
            z_input = z_mean.copy()
            z_input[j] = yi

            x_sample = vae.predict(type='decoder', z=[z_input], c=[c_mean])
            img = x_sample * 255. if size[-1] > 1 else x_sample
            figure[i * size[0]: (i + 1) * size[1], :, :] = img

        plt.figure(dpi=100, figsize=(10, 100))
        plt.imshow(figure.astype('int')) if size[-1] > 1 else plt.imshow(np.squeeze(figure), cmap='Greys_r')
        plt.savefig(save_dir + '/z_var%i.eps' % j)
        plt.close()


def plot_cvariation(x, vae, save_dir='./results'):
    # Predict
    z, _, _, c = vae.predict(type='encoder', x=x)

    # Compute
    c_mean = np.mean(c, axis=0)
    c_max = np.max(c, axis=0)
    c_min = np.min(c, axis=0)
    z_mean = np.mean(z, axis=0)

    # Plot
    n = x.shape[0]
    size = vae.input_shape
    for j in range(c_mean.size):
        grid_c = np.linspace(c_min[j], c_max[j], n)[::-1]
        figure = np.zeros((size[0] * n, size[1], size[-1]))
        for i, yi in enumerate(grid_c):
            c_input = c_mean.copy()
            c_input[j] = yi

            x_sample = vae.predict(type='decoder', z=[z_mean], c=[c_input])
            img = x_sample * 255. if size[-1] > 1 else x_sample
            figure[i * size[0]: (i + 1) * size[1], :, :] = img

        plt.figure(dpi=100, figsize=(10, 100))
        plt.imshow(figure.astype('int')) if size[-1] > 1 else plt.imshow(np.squeeze(figure), cmap='Greys_r')
        plt.savefig(save_dir + '/c_var%i.eps' % j)
        plt.close()


def plot_original(x, save_dir='./results'):
    # Plot
    n = 100
    size = x.shape[1:]
    figure = np.zeros((size[0] * n, size[1], size[-1]))
    for i in range(n):
        img = x[i] * 255. if size[-1] > 1 else x[i]
        figure[i * size[0]: (i + 1) * size[1], :, :] = img

    plt.figure(dpi=100, figsize=(10, 100))
    plt.imshow(figure.astype('int')) if size[-1] > 1 else plt.imshow(np.squeeze(figure), cmap='Greys_r')
    plt.savefig(save_dir + '/original.eps')
    plt.close()
