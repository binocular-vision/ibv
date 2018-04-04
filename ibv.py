from PIL import Image, ImageOps
import time
import json
import hashlib
import numpy as np
import random

from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.feature_extraction import image as skimage


def generate_gabor(size, shift, sigma, rotation, phase_shift, frequency):
    radius = (int((size[0]/2.0)), int((size[1]/2.0)))
    # a BUG is fixed in this line
    [x, y] = np.meshgrid(range(-radius[0], radius[0]),
                         range(-radius[1], radius[1]))
    x = x - int(shift[0])
    y = y - int(shift[1])
    x = x * frequency
    y = y * frequency
    tmp = x * np.cos(rotation) + y * np.sin(rotation) + phase_shift
    radius = (int(size[0]/2.0), int(size[1]/2.0))
    # a BUG is fixed in this line
    [x, y] = np.meshgrid(range(-radius[0], radius[0]),
                         range(-radius[1], radius[1]))

    x = x - int(shift[0])
    y = y - int(shift[1])
    x1 = x * np.cos(rotation) + y * np.sin(rotation)
    y1 = -x * np.sin(rotation) + y * np.cos(rotation)

    sinusoid = np.cos(tmp)

    gauss = np.e * \
        np.exp(np.negative(
            0.5 * ((x1**2 / sigma[0]**2) + (y1**2 / sigma[1]**2))))
    gauss = gauss / 2*np.pi * sigma[0] * sigma[1]

    gabor = gauss * sinusoid
    return gabor


def open_norm(path, verbose=False):
    raw = np.array(Image.open(path).convert("L"))
    norm = (raw - np.mean(raw)) / np.std(raw)

    if verbose:
        return raw, norm
    else:
        return norm


def linear_convolution(center, slide):
    if (center.shape != slide.shape):
        return
    padded_slide = np.zeros((center.shape[0], center.shape[1]*3))
    padded_slide[0:, center.shape[1]:center.shape[1]*2] = center
    # plt.imshow(padded_slide,origin="lower")
    # plt.show()
    estimate = np.zeros([center.shape[1]*2])
    for x in range(center.shape[1]*2):
        dot = np.sum(padded_slide[0:, 0+x:center.shape[1]+x] * slide)
        estimate[x] = dot
    # plt.plot(estimate)
    # plt.show()
    return np.abs(estimate)


def double_convolve(normal, shifted, image, pupillary_distance):

    # CHECKOUT https://github.com/maweigert/gputools
    # probably VERY advantageous to switch over to GPU for convolutions!

    normal_convolved = signal.convolve2d(
        image, normal, boundary='symm', mode='same')
    shifted_convolved = signal.convolve2d(
        image, shifted, boundary='symm', mode='same')

    return_shape = image.shape

    realigned = np.zeros(return_shape)

    normal_convolved = normal_convolved[0:, 0:-pupillary_distance]
    shifted_convolved = shifted_convolved[0:, pupillary_distance:]

    mul = normal_convolved * shifted_convolved
    # plt.imshow(mul,cmap="nipy_spectral")
    # plt.show()

    # REMOVE BELOW COMMENTS TO THRESH SUBHALF VALUES
    low_values_flags = mul < 0  # mul.max()*0.5  # Where values are low
    mul[low_values_flags] = 0  # All low values set to 0
    realigned[0:, pupillary_distance:] = mul
    return np.abs(mul)


def scale_disparity(activity_map, disparity_map):
    scaled_disparity = np.zeros(
        [activity_map.shape[0], activity_map.shape[1], disparity_map.shape[0]])
    scaled_disparity[:, :] = disparity_map
    for x in range(activity_map.shape[0]):
        for y in range(activity_map.shape[1]):
            scaled_disparity[x, y] = activity_map[x, y] * \
                scaled_disparity[x, y]

    return scaled_disparity



# In[4]:

def generate_patches(num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a):
    half_comp = patch_size**2
    patch_count = 0

    while (patch_count < num_patches):
        L = LGN(width=lgn_width, p=lgn_p, r=lgn_r, t=lgn_t, trans=lgn_a,
                make_wave=True, num_layers=2, random_seed=randint(1, 100))
        layer_activity = L.make_img_mat()
        patches_1 = np.array(skimage.extract_patches_2d(
            layer_activity[0], (patch_size, patch_size)))
        patches_2 = np.array(skimage.extract_patches_2d(
            layer_activity[1], (patch_size, patch_size)))
        reshaped_patches_1 = patches_1.reshape(-1,
                                               patches_1.shape[1]*patches_1.shape[1])
        reshaped_patches_2 = patches_2.reshape(-1,
                                               patches_2.shape[1]*patches_2.shape[1])
        composite_patches = np.concatenate(
            (reshaped_patches_1, reshaped_patches_2), axis=1)
        blacklist = []
        for x in range(composite_patches.shape[0]):
            if composite_patches[x][:half_comp].std() == 0.0 or composite_patches[x][half_comp:].std() == 0.0:
                blacklist.append(x)
        composite_patches = np.delete(
            composite_patches, np.array(blacklist), axis=0)
        if (patch_count == 0):
            patch_base = composite_patches
        else:
            patch_base = np.append(patch_base, composite_patches, axis=0)
        patch_count = patch_base.shape[0]

    return patch_base[:num_patches]


# In[5]:

def perform_ica(num_components, patches):
    # Run ICA on all the patches and return generated components
    # note, sensitive to n_components
    ica_instance = FastICA(n_components=num_components,
                           random_state=1, max_iter=1000000)
    icafit = ica_instance.fit(patches)
    ica_components = icafit.components_
    return ica_components


# In[6]:

def generate_filters(num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a):
    filter_count = 0
    while (filter_count < num_filters):
        patches = generate_patches(
            num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a)
        filters = perform_ica(num_components, patches)
        if (filter_count == 0):
            filter_base = filters
        else:
            filter_base = np.append(filter_base, filters, axis=0)
        filter_count = filter_base.shape[0]

    return filter_base[:num_filters]


# In[7]:

def unpack_filters(filters):
    half_filter = int(filters.shape[1]/2)
    filter_dim = int(np.sqrt(filters.shape[1]/2))
    first_eye = filters[:, 0:half_filter].reshape(-1, filter_dim, filter_dim)
    second_eye = filters[:, half_filter:].reshape(-1, filter_dim, filter_dim)
    return (first_eye, second_eye)


# In[8]:

def linear_disparity(first_eye, second_eye):
    disparity_map = np.empty([first_eye.shape[0], first_eye.shape[1]*2])
    for index in range(first_eye.shape[0]):
        disparity = linear_convolution(first_eye[index], second_eye[index])
        disparity_map[index] = disparity
    return disparity_map


# In[9]:

def normalize_disparity(disparity_map):
    with np.errstate(divide='ignore', invalid='ignore'):
        #normalize_disparity = (disparity_map - np.mean(disparity_map, axis=0)) / np.std(disparity_map)
        normalized_disparity = (disparity_map / np.mean(disparity_map, axis=0))

        #sum_normalized_disparity = np.sum(normalized_disparity, axis=0)
        #double_normalized_disparity = normalized_disparity / sum_normalized_disparity
    return normalized_disparity


# In[10]:

def generate_activity(autostereogram, asg_patch_size, first_eye, second_eye, disparity_map):
    for index in range(first_eye.shape[0]):
        # make this more elegant
        convolution = double_convolve(
            first_eye[index], second_eye[index], autostereogram, asg_patch_size)
        scaled_activity = scale_disparity(convolution, disparity_map[index])
        if index == 0:
            summed_activity = scaled_activity
        else:
            summed_activity = summed_activity + scaled_activity
    return summed_activity


# In[11]:

def estimate_depth(activity):
    depth_estimate = np.zeros([activity.shape[0], activity.shape[1]])
    for x in range(activity.shape[0]):
        for y in range(activity.shape[1]):
            peak = int(
                np.abs(np.nanargmax(activity[x, y])-int(activity.shape[2]/2)))
            #peak = np.nanargmax(activity[x,y])
            depth_estimate[x, y] = peak
    return depth_estimate


# In[12]:

def save_array(input_array, path):
    cast_array = (255.0 / input_array.max() *
                  (input_array - input_array.min())).astype(np.uint8)
    save_image = Image.fromarray(cast_array)
    colorized_image = ImageOps.colorize(save_image, (0, 0, 0), (0, 255, 0))
    colorized_image.save(path)
    print("SAVING ACTIVITY TO: %s" % (path))


# In[13]:

def generate_ident_hash(num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a, current_time):
    input_string = "%f%f%f%f%f%f%f%f%f%f" % (
        num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a, current_time)
    output_hash = hashlib.sha256(input_string.encode('utf-8')).hexdigest()
    return output_hash[:20]


# In[14]:

def calculate_optimal_p(t, r, a):
    p = t / (((np.pi * (r**2)/2))*(1+a))
    return p


# In[15]:

def disparity_distribution(disparity_map):
    dist = np.empty([disparity_map.shape[0]])
    for x in range(disparity_map.shape[0]):
        peak = np.abs(np.nanargmax(
            disparity_map[x])-int(disparity_map.shape[1]/2))
        dist[x] = int(peak)
    return dist


# In[16]:

def run_experiment(num_filters, num_components, num_patches, patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a, autostereogram, asg_patch_size, groundtruth, experiment_folder):
    filters = generate_filters(num_filters, num_components, num_patches,
                               patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a)
    split_filters = unpack_filters(filters)
    disparity_map = linear_disparity(split_filters[0], split_filters[1])

    # plt.hist(disparity_distribution(disparity_map))
    # plt.show()

    #normalized_disparity = disparity_map

    normalized_disparity = normalize_disparity(disparity_map)
    # plt.hist(disparity_distribution(normalized_disparity))
    # plt.show()

    activity = generate_activity(autostereogram, asg_patch_size,
                                 split_filters[0], split_filters[1], normalized_disparity)
    depth_estimate = estimate_depth(activity)
    correlation = np.corrcoef(depth_estimate.flatten(),
                              groundtruth.flatten())[0, 1]
    current_time = time.localtime()
    ident_hash = generate_ident_hash(num_filters, num_components, num_patches,
                                     patch_size, lgn_width, lgn_p, lgn_r, lgn_t, lgn_a, time.time())
    image_path = "%s/%s.png" % (experiment_folder, ident_hash)
    data_path = "%s/%s.json" % (experiment_folder, ident_hash)
    save_array(depth_estimate, image_path)
    params = {
        "num_filters": num_filters,
        "num_components": num_components,
        "num_patches": num_patches,
        "patch_size": patch_size,
        "lgn_width": lgn_width,
        "lgn_p": lgn_p,
        "lgn_r": lgn_r,
        "lgn_t": lgn_t,
        "lgn_a": lgn_a,
        "corr": np.abs(correlation),
        "time": time.strftime('%a, %d %b %Y %H:%M:%S GMT', current_time),
        "id": ident_hash
    }
    with open(data_path, 'w') as file:
        file.write(json.dumps(params))

    return params
