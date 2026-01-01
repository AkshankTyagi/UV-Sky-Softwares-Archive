# Functions to animate satellite in orbit and stars data
# Author: Ravi Ram

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.colors as mc
from matplotlib.figure import Figure


from configparser import ConfigParser

from star_spectrum import * 
from star_spectrum import GET_STAR_TEMP 
from Params_configparser import get_folder_loc

folder_loc, params_file = get_folder_loc()
# params_file = f'{folder_loc}init_parameter.txt'

def read_parameter_file(filename= params_file, param_set = 'Params_1'):
    config = ConfigParser()
    config.read(filename)
    global sat_name, Interval, spectra_width
    sat_name = config.get(param_set, 'sat_name')
    azm = float(config.get(param_set, 'azm'))
    ele = float(config.get(param_set, 'ele'))
    Interval = float(config.get(param_set, 'interval_bw_Frames'))
    spectra_width = config.get(param_set, 'longitudinal_spectral_width')
    return azm, ele

# main animate function
def animate(time_arr, state_vectors, celestial_coordinates, spectral_fov, r ):
    # init 3D earth and satellite view
    def init_orbit(ax):
        azm, ele = read_parameter_file()

        # set titles
        title = sat_name + ' Satellite position @ ' + time_arr[0].item().strftime('%Y-%m-%d - %H:%M:%S.')        
        ax.set_title(title)        
        
        # set labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        # set view
        ax.view_init(elev=ele, azim=azm)
        
        # set correct aspect ratio
        ax.set_box_aspect([1,1,1])
        set_axes_equal_3d(ax)
        
        # set limit
        size = 1.02
        limit = max(max(X), max(Y), max(Z))
        limit_low = min(min(X), min(Y), min(Z))
        ax.set_xlim(size*limit, -size*limit)
        ax.set_ylim(size*limit, -size*limit)
        ax.set_zlim(size*limit, -size*limit) 

        # earth
        ax.scatter(0, 0, 0, marker='o', c='deepskyblue', s=r)
        # satellite positions as a scatter plot
        satellite = ax.scatter(X[0], Y[0], Z[0], marker='o', c='k', s=2)
        # orbit path as a dotted line plot
        orbit = ax.plot(X[0], Y[0], Z[0], linewidth=0.9, linestyle='-.', c='k')[0] 

        # return
        return ax, satellite, orbit
    
    # init 2D sky view as seen in the velocity direction
    def init_sky(ax):
        global sky
               
        # set labels
        ax.set_xlabel('Right Ascension $^\circ$')
        ax.set_ylabel('Declination $^\circ$')
        
        # set titles
        ax.set_title('Sky view in the direction of velocity vector')
        
        # get initial frame celestial_coordinates data 
        P, S, Size = get_cles_data_by_frame(0, celestial_coordinates) 
        Size = Size[0]
        # print(Size)  
        # set axis limits
        ax.set_xlim(Size[0], Size[2])
        ax.set_ylim(Size[1], Size[3])  
        # ax.set_xlim(min(P[:,0]), max(P[:,0]))
        # ax.set_ylim(min(P[:,1]), max(P[:,1]))       
        
        # Scatter plot
        if (S[0] == 0.0001) :
            sky = ax.scatter(P[0], P[1], s=S[0], facecolors='White')
        else:
            sky = ax.scatter(P[:,0], P[:,1], s=S, facecolors='Red')
        
        # return
        return ax, sky    
    
    # init Intrinsic Spectral plot for all stars in the FOV
    # def init_Spectra(ax):
    #     global flux
        
    #     # set labels
    #     ax.set_xlabel('log$_{10}$[ Wavelength ($\AA$)]')
    #     ax.set_ylabel('log$_{10}$[ Flux (FLAM)] + offset')
        
    #     # set titles
    #     ax.set_title('Spectrum of the stars in the Sky view')

    #     X_wavelength, Y_Spectra_per_star, ra, dec = get_spectral_data_by_frame(0, spectra)
    #     _, _, Size = get_cles_data_by_frame(0, celestial_coordinates) 
    #     Size = Size[0]

    #     if (X_wavelength[0]!=0):
    #         # y_offset = [float(float(d) - Size[1]) * (Size[3] - Size[1]) for d in np.array(dec)]
    #         # flux = ax4.plot(np.log10(X_wavelength), np.log10(Y_Spectra_per_star[0]), label = f'{ra[0]},{dec[0]}')
    #         for i in range(len(Y_Spectra_per_star)):
    #             y_offset = (dec[i] - Size[1])*(Size[3] - Size[1]) 
    #             # flux = ax.plot(np.log10(X_wavelength), Y_Spectra_per_star[i], label = f'ra: {ra[i]}  ; dec: {dec[i]}') #+ [y_offset]*len(Y_Spectra_per_star[i])
    #             flux = ax.plot(np.log10(X_wavelength),  np.log10(Y_Spectra_per_star[i]), label = f'ra: {ra[i]}  ; dec: {dec[i]}') # + [y_offset]*len(Y_Spectra_per_star[i])
    #             ax.set_ylim(-1, 11)
    #     else:
    #         wavelengths = np.linspace(100, 4000, 1000)
    #         y_zeros = np.zeros_like(wavelengths)
    #         flux= ax.plot(np.log10(wavelengths), y_zeros, color='gray', linestyle='--', label='y = 0')
    #         ax.set_ylim(-1, 11)
    #     ax.legend()
    #     # ax.clear()

        return ax, flux
    
    # init Absorption Spectra plot for all stars in the FOV
    def init_spectra(ax):
        global spectra
        global colors,flux_cmap
        X_wavelength, Y_photons_per_star, ra, dec = get_photons_data_by_frame(0, spectral_fov)
        
        # set titles
        ax.suptitle('Absorption Spectra of each star')
    
        if (X_wavelength[0]!=0): #checks for stars in the field of view
            n = len(ra)
            ax_pos=[]
            wave_min = min(X_wavelength)
            wave_max = max(X_wavelength)
            photons_max =0
            for i in range(n):
                MaxP = max(Y_photons_per_star[i])
                if (MaxP >photons_max):
                    photons_max = MaxP
            color_data = (X_wavelength - wave_min)/(wave_max- wave_min)

            # Create a custom colormap based only on Violet-Indigo-Blue from spectrum
            colors = plt.cm.rainbow(np.linspace(0, 1,  1500))
            colors = colors[0:int(len(X_wavelength))]

            if n == 1:  # only 1 star in the Fov
                spectra = ax.add_subplot( )
                ax_pos = spectra
                # photons_max = max(Y_photons_per_star[0])
                #calculate alpha value from stars flux of photons, for each wavelength
                alpha_val = calc_inv_opacity(Y_photons_per_star[0],len(Y_photons_per_star[0]), photons_max)  
                
                # Calculates observance of a wavelength by its flux of photons
                colors2 = calc_obs_color(colors, alpha_val, len(X_wavelength))
                flux_cmap = mc.ListedColormap(colors2)

                spectra.imshow(color_data[:].reshape(1, -1), cmap=flux_cmap, aspect='auto', extent=(wave_min, wave_max, 0,0.1))
                spectra.set_xlabel(' Wavelength ($\AA$)')
                spectra.set_ylabel(f'Star {i+1}')
            else:
                spectra = ax5.subplots(n, 1, sharex=True)

                for i, axs in enumerate(spectra): #plot spectra for each star 1 by 1
                    # photons_max = max(Y_photons_per_star[i])
                    alpha_val = calc_inv_opacity(Y_photons_per_star[i],len(Y_photons_per_star[i]), photons_max)
                    colors2 = calc_obs_color(colors, alpha_val, len(X_wavelength))
                    flux_cmap = mc.ListedColormap(colors2)                    

                    spectra[i] =axs.imshow(color_data[:].reshape(1, -1), cmap=flux_cmap, aspect='auto', extent=(wave_min, wave_max, 0,0.1))
                    if (i == n-1):
                        axs.set_xlabel(' Wavelength ($\AA$)')
                    axs.set_ylabel(f'Star {i+1}: RA= {ra[i]}')
                    ax_pos.append(axs)
            # Add colorbar
            # norm = mc.Normalize(vmin=wave_min, vmax=wave_max)
            # scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=mc.ListedColormap(colors))
            # scalar_mappable.set_array([])  # Optional: set an empty array to the ScalarMappable
            # ax.colorbar(scalar_mappable, orientation='horizontal', ax = ax_pos, label='Wavelength ($\AA$)')

        else:
            spectra = ax5.add_subplot( )
            colors2 = np.zeros((len(X_wavelength), 4))
            for j in range(len(X_wavelength)): 
                colors2[j][3] = 1 
            flux_cmap = mc.ListedColormap(colors2)
            color_data = np.zeros(len(X_wavelength))
            
            spectra.imshow(color_data[:].reshape(1, -1), cmap=flux_cmap, aspect='auto', extent=(10, 3800, 0, 1))
            spectra.set_ylim(0, 1)  # Set y-axis limits to 0 and 1
            spectra.set_xlabel('wavelength')
            spectra.set_ylabel(f'No Star present')
            # Add colorbar
            # norm = mc.Normalize(vmin=100, vmax=3800)
            # scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=flux_cmap)
            # scalar_mappable.set_array([])  # Optional: set an empty array to the ScalarMappable
            # ax.colorbar(scalar_mappable,orientation='horizontal', ax = spectra, label='Wavelength ($\AA$)')

        return ax, spectra


    # init # of Photons plot
    def init_photons(ax):
        global phots
        
        # set labels
        ax.set_xlabel('Wavelength ($\AA$)')
        ax.set_ylabel('Number of Photons')
        
        # set titles
        ax.set_title('# of Photons from the stars in the Sky view')

        X_wavelength, Y_photons_per_star, ra, dec = get_photons_data_by_frame(0, spectral_fov)
        # _, _, Size = get_cles_data_by_frame(0, celestial_coordinates)  #used in y_offset
        # Size = Size[0]
        
        if (X_wavelength[0]!=0):
            wave_min = min(X_wavelength)
            wave_max = max(X_wavelength)
            # y_offset = [float(float(d) - Size[1]) * (Size[3] - Size[1]) for d in np.array(dec)]
            # flux = ax4.plot(np.log10(X_wavelength), np.log10(Y_Spectra_per_star[0]), label = f'{ra[0]},{dec[0]}')
            for i in range(len(Y_photons_per_star)):
                # y_offset = (dec[i] - Size[1])*(Size[3] - Size[1]) 
                # phots = ax.plot(np.log10(X_wavelength), Y_Spectra_per_star[i], label = f'ra: {ra[i]}  ; dec: {dec[i]}') #+ [y_offset]*len(Y_Spectra_per_star[i])
                phots = ax.plot(X_wavelength,  Y_photons_per_star[i], label = f'ra: {ra[i]}  ; dec: {dec[i]}') # + [y_offset]*len(Y_Spectra_per_star[i])
                ax.set_xlim(wave_min, wave_max)
        else:
            wavelengths = np.linspace(100, 3800, 1000)
            y_zeros = np.zeros_like(wavelengths) 
            phots= ax.plot(wavelengths, y_zeros, color='gray', linestyle='--', label='y = 0')
            ax.set_xlim(min(wavelengths), max(wavelengths))
        ax.legend()
        # ax.clear()

        return ax, phots
    

    # initialize plot
    def init():
        global fig, ax2, ax3, ax4, ax5
        global orbit, satellite, sky,  phots, spectra
        global X, Y, Z
        global RA, DEC
        
        # position vectors
        X, Y, Z = state_vectors[0], state_vectors[1], state_vectors[2]
        # Sent for figure
        font = {'size'   : 6}
        plt.rc('font', **font)
            
        # Create 2x2 sub plots
        gs = gridspec.GridSpec(2, 2, wspace=0.5, width_ratios=[1, 2], ) # , hspace=0.5, , width_ratios=[1, 2]
        # fig and ax
        fig = plt.figure(layout='constrained', figsize=(12,6)) # figsize=(8,6)
        subfigs = fig.subfigures(2, 2, wspace=0.07, hspace= 0.03, width_ratios=[1, 2], height_ratios= [1, 1]) #, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})

        # fig = plt.figure(layout='constrained', figsize=(10, 4))
        # subfigs = fig.subfigures(1, 2, wspace=0.07)
        # print(np.shape(subfigs))

        # axsLeft = subfigs[0].subplots(1, 2, sharey=True)

        # row 0, col 0
        ax2 = subfigs[0,0].add_subplot( projection='3d')
        # ax2 = fig.add_subplot(gs[0, 0], projection='3d' )
        # set layout
        ax2, satellite, orbit = init_orbit(ax2)  

        # row 0, col 1
        ax3 = subfigs[1,0].add_subplot( facecolor="black", aspect= 0.15)
        # ax3 = fig.add_subplot(gs[1, 0], facecolor="black", aspect= 0.15 )

        # initialize sky
        ax3, sky = init_sky(ax3)

        # # row 1, col 0
        # ax4 = fig.add_subplot(gs[1, 0])
        # # initialize Spectrum Plot
        # ax4, flux = init_Spectra(ax4)

        # row 1, col 0
    
        ax4 = subfigs[0,1].add_subplot()
        # ax4.set_aspect('equal', adjustable='box') 
        # ax4 = fig.add_subplot(gs[0, 1])
        # initialize Photons Plot
        ax4, phots = init_photons(ax4)

        # row 1, col 1
        ax5 = subfigs[1,1]
        # ax5.set_size_inches(7, 3)
        # initialize Photons Plot
        ax5, spectra = init_spectra(ax5)

        # to avoid subplot title overlap with x-tick
        # fig.tight_layout()
        
        # return
        return fig, satellite, orbit, sky, phots, spectra

    def update(i, satellite, orbit, sky, phots, spectra):
        # stack as np columns for scatter plot
        xyi, xi, yi, zi = get_pos_data_by_frame(i)
        # print ('frame number',i+1,'- satellite path:', xi, yi, zi)
        title = sat_name+' Satellite position @ ' + time_arr[i].item().strftime('%Y-%m-%d - %H:%M:%S.')        
        ax2.set_title(title)
        # print("update animation")
        # _offsets3d for scatter
        satellite._offsets3d = ( xi, yi, zi )
        # .set_data() for plot...
        orbit.set_data(xi, yi)
        orbit.set_3d_properties(zi)
        
        # get frame data. pos[ra, dec], size
        P, S, Size = get_cles_data_by_frame(i, celestial_coordinates)
        Size = Size[0]
        # print(S)
        # Update scatter object
        sky.set_offsets(P)
        # print('P is working')
        sky.set_sizes(S)
        # print('S is working')   

        # change sky limits
        ax3.set_xlim(Size[0], Size[2])
        ax3.set_ylim(Size[1], Size[3])    
        # ax3.set_xlim(min(P[:,0]), max(P[:,0]))
        # ax3.set_ylim(min(P[:,1]), max(P[:,1]))
        
        # get updated photons data by frame
        X_wavelength, Y_photons_per_star, ra, dec = get_photons_data_by_frame(i, spectral_fov)
        if (X_wavelength[0]!=0):
            ax4.clear()
            wave_min = min(X_wavelength)
            wave_max = max(X_wavelength)
            # y_offset = [float(float(d) - Size[1]) * (Size[3] - Size[1]) for d in np.array(dec)]
            # flux = ax4.plot(np.log10(X_wavelength), np.log10(Y_Spectra_per_star[0]), label = f'{ra[0]},{dec[0]}')
            for k in range(len(Y_photons_per_star)):
                # y_offset = (dec[i] - Size[1])*(Size[3] - Size[1]) 
                # phots = ax4.plot(np.log10(X_wavelength), Y_Spectra_per_star[i], label = f'ra: {ra[i]}  ; dec: {dec[i]}') #+ [y_offset]*len(Y_Spectra_per_star[i])
                phots = ax4.plot(X_wavelength,  Y_photons_per_star[k], label = f'ra: {ra[k]}  ; dec: {dec[k]}') # + [y_offset]*len(Y_Spectra_per_star[i])
                ax4.set_xlim(wave_min, wave_max)
        else:
            ax4.clear()
            wavelengths = np.linspace(100, 3800, 1000)
            y_zeros = np.zeros_like(wavelengths) 
            phots= ax4.plot(wavelengths, y_zeros, color='gray', linestyle='--', label='y = 0')
            ax4.set_xlim(min(wavelengths), max(wavelengths))
            
        ax4.legend()
        # set labels
        ax4.set_xlabel('Wavelength ($\AA$)')
        ax4.set_ylabel('Number of Photons')
        # set titles
        ax4.set_title('# of Photons from the stars in the Sky view')


        # setting up the absorption spectra plots
        ax5.clear()
        ax5.suptitle('Absorption Spectra of each star')

        if (X_wavelength[0]!=0): #checks for stars in the field of view
            n = len(ra)
            ax_pos =[]
            wave_min = min(X_wavelength)
            wave_max = max(X_wavelength)
            photons_max =0
            for i in range(n):
                MaxP = max(Y_photons_per_star[i])
                if (MaxP >photons_max):
                    photons_max = MaxP
            print (photons_max)
            color_data = (X_wavelength - wave_min)/(wave_max- wave_min)

            # Create a custom colormap based only on Violet-Indigo-Blue from spectrum
            colors = plt.cm.rainbow(np.linspace(0, 1,  1500))
            colors = colors[0:int(len(X_wavelength))]

            if n == 1:  # only 1 star in the Fov
                spectra = ax5.add_subplot( )
                ax_pos = spectra
                # photons_max = max(Y_photons_per_star[0])
                alpha_val = calc_inv_opacity(Y_photons_per_star[0],len(Y_photons_per_star[0]), photons_max)
                colors2 = calc_obs_color(colors, alpha_val, len(X_wavelength))
                flux_cmap = mc.ListedColormap(colors2)
                
                spectra.imshow(color_data[:].reshape(1, -1), cmap=flux_cmap, aspect='auto', extent=(wave_min, wave_max, 0,0.1))
                spectra.set_xlabel(' Wavelength ($\AA$)')
                spectra.set_ylabel(f'Star 1, RA = {ra[0]}')
            else:
                spectra = ax5.subplots(n, 1, sharex=True)
                for i, axs in enumerate(spectra): #plot spectra for each star 1 by 1
                    # photons_max = max(Y_photons_per_star[i])
                    alpha_val = calc_inv_opacity(Y_photons_per_star[i],len(Y_photons_per_star[i]), photons_max)
                    colors2 = calc_obs_color(colors, alpha_val, len(X_wavelength))
                    flux_cmap = mc.ListedColormap(colors2)

                    spectra[i] = axs.imshow(color_data[:].reshape(1, -1), cmap=flux_cmap, aspect='auto', extent=(wave_min, wave_max, 0,0.1))
                    if (i == n-1):
                        axs.set_xlabel(' Wavelength ($\AA$)')
                    axs.set_ylabel(f'Star {i+1},  RA = {ra[i]}')
                    ax_pos.append(axs)
            #add colorbar
            # norm = mc.Normalize(vmin=wave_min, vmax=wave_max)
            # scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=mc.ListedColormap(colors))
            # scalar_mappable.set_array([])  # Optional: set an empty array to the ScalarMappable
            # ax5.colorbar(scalar_mappable,orientation='horizontal', ax = ax_pos, label='Wavelength ($\AA$)')
        else:
            spectra = ax5.add_subplot( )

            colors2 = np.zeros((len(X_wavelength), 4))
            for j in range(len(X_wavelength)): 
                colors2[j][3] = 1 
            flux_cmap = mc.ListedColormap(colors2)

            color_data = np.zeros(len(X_wavelength))
            spectra.imshow(color_data[:].reshape(1, -1), cmap=flux_cmap, aspect='auto', extent=(10, 3800, 0, 1))
            spectra.set_xlabel('wavelength')
            spectra.set_ylim(0, 1)  # Set y-axis limits to 0 and 1
            spectra.set_ylabel(f'No Star present')

            # norm = mc.Normalize(vmin=10, vmax=3800)
            # scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=flux_cmap)
            # scalar_mappable.set_array([])  # Optional: set an empty array to the ScalarMappable
            # ax5.colorbar(scalar_mappable, orientation='horizontal', ax = spectra, label='Wavelength ($\AA$)')

        # return
        return satellite, orbit, sky, phots, spectra

    # run animation
    def run():
        # plot init
        fig, satellite, orbit, sky, phots, spectra = init()
        # total no of frames
        frame_count = len(X)
        # print (frame_count)
        # create animation using the animate() function
        ani = animation.FuncAnimation(fig, update,
                                      frames=frame_count, interval= Interval, 
                                      fargs=(satellite, orbit, sky, phots, spectra ),
                                      blit=False, repeat=False)
        # save
        plt.show()
        print("animation complete")
        # ani.save('satellite.gif', writer="ffmpeg")
        # print("saved")
        # show
        return ani
    
    # run animation
    run()

    # end-plot-sky
    return

# Set 3D plot axes to equal scale. 
# Required since `ax.axis('equal')` and `ax.set_aspect('equal')` don't work on 3D.
# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal_3d(ax: plt.Axes):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)
    return

# set axis limits
def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
    return

# get satellite position data for the given index
def get_pos_data_by_frame(i):
    # pack it like thisfor set_3d_properties   
    xi, yi, zi = X[..., :i+1],  Y[..., :i+1], Z[..., :i+1]
    xy = np.column_stack((xi, yi))
    # return
    return xy, xi, yi, zi

# get celestial coordinates of the stars for the given index
def get_cles_data_by_frame(i, data):
    # select a frame
    # print(data[i])
    frame, d, frame_boundary = zip(data[i])
    # print (frame_boundary)
    # slice into columns
    if d[0]:
        c = list(zip(*d[0]))
        print('Frame',frame[0]+1,'has', len(c[0]),'stars.' )
        # pack it
        #ra, dec, size = np.array(c[0]), np.array(c[1]), np.array(c[2])
        ra, dec, size = c[0], c[1], c[2]
        # stack as np columns for scatter plot
        cles_pos = np.column_stack((ra, dec))
        # Print Stellar Data of the stars in the FOV
        hip, mag, parallax, B_V, Spectral_type = c[3], c[4], c[5], c[6], c[7]
        if (len(c[0])>1):
            print('  The stars in the FOV are:')
            for i in range(len(c[0])):     
                Temp = GET_STAR_TEMP(str(Spectral_type[i]))
                print( f"{str(i+1)}) Hipp_number= {str(hip[i])}; Ra & Dec: {str(ra[i])} {str(dec[i])}; Johnson Mag= {str(mag[i])}; trig Parallax= {str(parallax[i])}; Color(B-V)= {str(B_V[i])}; Spectral_Type: {str(Spectral_type[i])}; T_index: {Temp}" , end="\n")

        else:
            print('  The star in the FOV is:')
            Temp = GET_STAR_TEMP(str(Spectral_type[0]))
            print( f"  Hipp_number= {str(hip[0])}; Ra & Dec: {str(ra[0])} {str(dec[0])}; Johnson Mag= {str(mag[0])}; trig Parallax= {str(parallax[0])}; Color(B-V)= {str(B_V[0])}; Spectral_Type: {str(Spectral_type[0])}; T_index: {Temp}", end="\n")

        # return
        return cles_pos, size, frame_boundary 
    else:
        print('Frame',frame[0]+1,'is EMPTY', end="\n")
        no_star = [0,0]
        zero_size =(0.0001,)
        return no_star, zero_size, frame_boundary

# get alpha values or inverse opacity 
def calc_inv_opacity(data, Range, max_val): 
    alpha_val = np.zeros(Range)
    for j in range(Range):
        if (data[j] <= 1):
            alpha_val[j] = (1)
        elif (data[j] >= 0.5 * max_val):
            alpha_val[j] = (max_val - data[j])/(5*max_val)
        elif ( data[j] >= 0.01 * max_val):
            alpha_val[j] = (max_val - data[j])/(4*max_val)
        elif (data[j] >= 0.001 * max_val):
            alpha_val[j] = (max_val - data[j])/(2*max_val)
        elif ( data[j] >= 0.00001 * max_val):
            alpha_val[j] = 0.65
        elif ( data[j] < 0.00001 * max_val):
            alpha_val[j] = 0.7

    return alpha_val

# get observed intensity of photons at each wavelength and convert to RGBA values for cmap 
def calc_obs_color(colors, alpha_val, Range):
    colors2 = np.zeros((Range, 4))
    for j in range(Range): 
        colors2[j][0] = colors[j][0] - (colors[j][0]*alpha_val[j])
        colors2[j][1] = colors[j][1] - (colors[j][1]*alpha_val[j])
        colors2[j][2] = colors[j][2] - (colors[j][2]*alpha_val[j])
        colors2[j][3] = 1 
    return colors2

# make star spectra graph data from photons data
# def get_color_data(data, wavelength, photons_data, dec):
#     spectra_width
#     for i in range(len(dec)):
#         row_num = 0
#         for row in data:
#             if (row_num>= int(dec[i])- spectra_width/2) and (row_num<= int(dec[i])- spectra_width/2) :
                
#             row_num = row_num + 1
#     return data

# get Spectral data of the stars for the given frame index 
def get_spectral_data_by_frame(i, spectral_FOV):
    frame = spectral_FOV.frame[i], 
    Wavelength = spectral_FOV.wavelength[i]
    Spectra_per_star = spectral_FOV.spectra_per_star[i]
    ra = spectral_FOV.ra[i]
    dec = spectral_FOV.dec[i]

    return Wavelength, Spectra_per_star,ra, dec

# get photon number data of the stars for the given frame index 
def get_photons_data_by_frame(i, spectral_FOV):

    frame = spectral_FOV.frame[i], 
    Wavelength = spectral_FOV.wavelength[i]
    photon_per_star = spectral_FOV.photons[i]
    ra = spectral_FOV.ra[i]
    dec = spectral_FOV.dec[i]

    return Wavelength, photon_per_star,ra, dec


        # # get updated spectral data by frame
        # X_wavelength, Y_Spectra_per_star, ra, dec = get_spectral_data_by_frame(i, spectra)

        # # Update Spectral plot
        # if (X_wavelength[0]!=0):
        #     ax4.clear()
        #     # y_offset = [float(float(d) - Size[1]) * (Size[3] - Size[1]) for d in np.array(dec)]
        #     # flux = ax4.plot(np.log10(X_wavelength), np.log10(Y_Spectra_per_star[0]), label = f'{ra[0]},{dec[0]}')
        #     for j in range(len(Y_Spectra_per_star)):
        #         # y_offset = (dec[i] - Size[1])*(Size[3] - Size[1])
        #         # flux = ax4.plot(np.log10(X_wavelength), Y_Spectra_per_star[i], label = f'ra: {ra[i]}  ; dec: {dec[i]}') #+ [y_offset]*len(Y_Spectra_per_star[i])
        #         flux = ax4.plot(np.log10(X_wavelength), np.log10(Y_Spectra_per_star[j]), label = f'ra: {ra[j]}  ; dec: {dec[j]}') #+ [y_offset]*len(Y_Spectra_per_star[i])
        #         ax4.set_ylim(-1, 11)        
        # else:
        #     ax4.clear()
        #     wavelengths = np.linspace(100, 4000, 1000)
        #     y_zeros = np.zeros_like(wavelengths)
        #     flux= ax4.plot(np.log10(wavelengths), y_zeros, color='gray', linestyle='--', label='y = 0')
        #     ax4.set_ylim(-1, 11)
        # ax4.legend()
        # # set labels
        # ax4.set_xlabel('log$_{10}$[ Wavelength ($\AA$)]')
        # ax4.set_ylabel('log$_{10}$[ Flux (FLAM)] + offset')
        # # set title
        # ax4.set_title('Spectrum of the stars in the Sky view')
