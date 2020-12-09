d2r = np.pi/180
r2d = 180/np.pi
eps = 1.e-6

# @as_op(itypes=[T.dvector, T.dvector, T.dscalar], otypes=[T.dvector])
def rotate(pole, rotation_pole, angle):
    # The idea is to rotate the pole so that the Euler pole is
    # at the pole of the coordinate system, then perform the
    # requested rotation, then restore things to the original
    # orientation
    carttodir = cartesian_to_spherical(rotation_pole)
    lon, lat, intensity = carttodir[0], carttodir[1], carttodir[2]

    colat = 90. - lat
    m1 = construct_euler_rotation_matrix(-lon
                                         * d2r, -colat * d2r, angle * d2r)
    m2 = construct_euler_rotation_matrix(0., colat * d2r, lon * d2r)
    return np.dot(m2, np.dot(m1, pole))


def construct_euler_rotation_matrix(alpha, beta, gamma):
    """
    Make a 3x3 matrix which represents a rigid body rotation,
    with alpha being the first rotation about the z axis,
    beta being the second rotation about the y axis, and
    gamma being the third rotation about the z axis.

    All angles are assumed to be in radians
    """
    rot_alpha = np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                          [np.sin(alpha), np.cos(alpha), 0.],
                          [0., 0., 1.]])
    rot_beta = np.array([[np.cos(beta), 0., np.sin(beta)],
                         [0., 1., 0.],
                         [-np.sin(beta), 0., np.cos(beta)]])
    rot_gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0.],
                          [np.sin(gamma), np.cos(gamma), 0.],
                          [0., 0., 1.]])
    rot = np.dot(rot_gamma, np.dot(rot_beta, rot_alpha))
    return rot

def spherical_to_cartesian(longitude, latitude, norm = 1):
    #    assert(np.all(longitude >= 0.) and np.all(longitude <= 360.))
    #    assert(np.all(latitude >= -90.) and np.all(latitude <= 90.))
    #    assert(np.all(norm >= 0.))
    colatitude = 90. - latitude
    return np.array([norm * np.sin(colatitude * d2r) * np.cos(longitude * d2r),
                     norm * np.sin(colatitude * d2r) * np.sin(longitude * d2r),
                     norm * np.cos(colatitude * d2r)])

def cartesian_to_spherical(vecs):
    v = np.reshape(vecs, (3, -1))
    norm = np.sqrt(v[0, :] * v[0, :] + v[1, :] * v[1, :] + v[2, :] * v[2, :])
    latitude = 90. - np.arccos(v[2, :] / norm) * r2d
    longitude = np.arctan2(v[1, :],v[0, :]) * r2d

    return longitude, latitude, norm

# now the longitude will be constrained to be within 0 to 360 degrees
def clamp_longitude( lons ):
    lons = np.asarray(lons)
    lons = np.fmod(lons, 360.)
    lons[np.where(lons < 0)] += 360.
    return lons

def two_sigma_from_kappa(kappa):
    return 140. / np.sqrt(kappa)


def kappa_from_two_sigma(two_sigma):
    return 140. * 140. / two_sigma / two_sigma

class APWP(object):

    def __init__(self, name, paleomagnetic_pole_list, n_euler_poles, sample_size = 2000):
        for p in paleomagnetic_pole_list:
            assert (isinstance(p, PaleomagneticPole))

        self._name = name
        self._poles = paleomagnetic_pole_list
        self.n_euler_rotations = n_euler_poles

        self._age_list = [p._age for p in self._poles]
        self._start_age = max(self._age_list)
        self._start_pole = self._poles[np.argmax(self._age_list)]
        self._sample_size = sample_size

    def create_model(self, site_lon_lat=[1., 0.], k=1., kw = -1., euler_rate=1., tpw_rate_scale = None):
        assert euler_rate > 0.0, "rate_scale must be a positive number."
        assert tpw_rate_scale == None or tpw_rate_scale > 0.0
        assert kw <= 0.0, "Nonnegative Watson concentration parameters are not supported."
        if tpw_rate_scale is None:
            self.include_tpw = False
        else:
            self.include_tpw = True

        with pm.Model() as APWP_model:
            start = VMF('start',
                        lon_lat=[self._start_pole.longitude, self._start_pole.latitude],
                        k=kappa_from_two_sigma(self._start_pole._A95),
                        testval = np.array([1., 0.]), shape = 2)

#             if self.n_euler_rotations == 0 and tpw_rate_scale!= None:
#                 tpw_pole_angle = pm.Uniform('tpw_pole_angle', 0., 360., value=0., observed=False)
#                 tpw_rate = pm.Exponential('tpw_rate', tpw_rate_scale)

#             elif self.n_euler_rotations == 1:
#                 euler_1 = VMF('euler_1', lon_lat = [1., 0.], k = k, testval = np.array([1., 0.]), shape = 2)
#                 rate_1 = pm.Exponential('rate_1', euler_rate)

#                 for i in range(len(self._poles)):
#                     p = self._poles[i]

#                     if p._age_type == 'gaussian':
#                         pole_age = pm.Normal('t'+str(i), mu=self._age_list[i], tau=1/(p._sigma_age**-2))
#                     elif p._age_type == 'uniform':
#                         pole_age = pm.Uniform('t'+str(i), lower=p._sigma_age[0], upper=p._sigma_age[1])

#                     lon_lat = pole_position_1e(start, euler_1, rate_1, pole_age )
#                     observed_pole = VMF('p'+str(i), lon_lat, k = kappa_from_two_sigma(p._A95), observed=[p.longitude, p.latitude])

#             elif self.n_euler_rotations == 2:

            euler_1 = VMF('euler_1', lon_lat = site_lon_lat, k = k, testval = np.array([1., 0.]), shape = 2)
            rate_1 = pm.Exponential('rate_1', euler_rate)
            euler_2 = VMF('euler_2', lon_lat = site_lon_lat, k = k, testval = np.array([1., 0.]), shape = 2)
            rate_2 = pm.Exponential('rate_2', euler_rate)
            switchpoint = pm.Uniform('switchpoint', min(self._age_list), max(self._age_list))

            for i in range(len(self._poles)):
                p = self._poles[i]

                if p._age_type == 'gaussian':
                    pole_age = pm.Normal('t'+str(i), mu=self._age_list[i], tau=1/(p._sigma_age**-2))
                elif p._age_type == 'uniform':
                    pole_age = pm.Uniform('t'+str(i), lower=p._sigma_age[0], upper=p._sigma_age[1])

                lon_lat = pole_position(start, euler_1, rate_1, euler_2, rate_2, switchpoint, pole_age )
                observed_pole = VMF('p'+str(i), lon_lat, k = kappa_from_two_sigma(p._A95), observed=[p.longitude, p.latitude])

            trace = pm.sample(self._sample_size, step = pm.Metropolis())
#             for i in range(len(self._poles)):
#                 p = self._poles[i]

# #                 time = self._start_age
#                 if p._age_type == 'gaussian':
#                     pole_age = pm.Normal('t'+str(i), mu=self._age_list[i], tau=1/(p._sigma_age**-2))
#                 elif p._age_type == 'uniform':
#                     pole_age = pm.Uniform('t'+str(i), lower=p._sigma_age[0], upper=p._sigma_age[1])

#                 if self.n_euler_rotations == 2:
#                     lon_lat = pole_position(start, eulers[0], rates[0], eulers[1], rates[1], changes[0], pole_age)
#                 lon_lat = pole_position(start, euler_1, rate_1, euler_2, rate_2, switchpoint, pole_age )
#                 observed_pole = VMF('p'+str(i), lon_lat, k=kappa_from_two_sigma(p._A95), observed=[p.longitude, p.latitude])
#                 observed_pole = VMF('p'+str(i), lon_lat, k = kappa_from_two_sigma(p._A95), observed=[p.longitude, p.latitude])
#             if self.include_tpw:
#                 tpw_pole_angle = pm.Uniform('tpw_pole_angle', lower = 0., upper = 360.)
#                 tpw_rate = pm.Exponential('tpw_rate', tpw_rate_scale)

        return trace



# based on poles.py by I. Rose

class Pole(object):
    """
    Class representing a pole on the globe:
    essentially a 3-vector with some additional
    properties and operations.
    """

    def __init__(self, longitude, latitude, magnitude = 1, A95=None):
        """
        Initialize the pole with lon, lat, and A95 uncertainty. Removed norm from Rose version, here we assume everything is unit vector.
        longitude, latitude, and A95 are all taken in degrees.
        """
        self.longitude = longitude
        self.latitude = latitude
        self.colatitude = 90 - latitude
        self.magnitude = magnitude
        self._pole = spherical_to_cartesian(self.longitude, self.latitude, self.magnitude) # pole position in cartesian coordinates, easier for addition operations
        self._A95 = A95

#     @property
#     def longitude(self):
#         return np.arctan2(self._pole[1], self._pole[0]) * rot.r2d

#     @property
#     def latitude(self):
#         return 90. - np.arccos(self._pole[2] / self.norm) * rot.r2d

#     @property
#     def colatitude(self):
#         return np.arccos(self._pole[2] / self.norm) * rot.r2d

#     @property
#     def norm(self):
#         return np.sqrt(self._pole[0] * self._pole[0] + self._pole[1] * self._pole[1] + self._pole[2] * self._pole[2])

#     @property
#     def angular_error(self):
#         return self._angular_error

    def copy(self):
        return copy.deepcopy(self)

    def rotate(self, pole, angle):
        # The idea is to rotate the pole about a given pole
        # at the pole of the coordinate system, then perform the
        # requested rotation, then restore things to the original
        # orientation
        p = pole._pole

        lon, lat, intensity = cartesian_to_spherical(p)
        colat = 90. - lat
        m1 = construct_euler_rotation_matrix(-lon * d2r, -colat * d2r, angle * d2r)
        m2 = construct_euler_rotation_matrix(0., colat * d2r, lon * d2r)
        self._pole = np.dot(m2, np.dot(m1, self._pole))
        self.longitude = cartesian_to_spherical(self._pole.tolist())[0].tolist()[0]
        self.latitude = cartesian_to_spherical(self._pole.tolist())[1].tolist()[0]
        self._pole = spherical_to_cartesian(self.longitude, self.latitude, self.magnitude)
        self.colatitude = 90 - self.latitude

    def _rotate(self, pole, angle):
        print(self.longitude, self.latitude)
        p = pole._pole

        lon, lat, intensity = cartesian_to_spherical(p)
        lon = T.as_tensor_variable(lon[0])
        lat = T.as_tensor_variable(lat[0])

        colat = 90. - lat
        m1 = construct_euler_rotation_matrix(-lon * d2r, -colat * d2r, angle * d2r)
        m2 = construct_euler_rotation_matrix(0., colat * d2r, lon * d2r)
        self._pole = np.dot(m2, np.dot(m1, self._pole))
#         return np.array(cartesian_to_spherical(self._pole.tolist())[0],
#                         cartesian_to_spherical(self._pole.tolist())[1])
        self.longitude = cartesian_to_spherical(self._pole.tolist())[0].tolist()[0]
        self.latitude = cartesian_to_spherical(self._pole.tolist())[1].tolist()[0]
        self._pole = spherical_to_cartesian(self.longitude, self.latitude, self.magnitude)
#         print(self.longitude)
#         print(self.latitude)
#         print(self._pole)
        print(self.longitude, self.latitude)
        self.colatitude = 90 - self.latitude

    def add(self, pole):
        self._pole = self._pole + pole._pole

    # 4/9/2020 YZ corrected the transform for the circle path from PlateCarree to Geodetic so that the poles and circles won't look odd at high latitudes.
    def plot(self, axes, south_pole=False, **kwargs):
        artists = []
        if self._A95 is not None:
            lons = np.linspace(0, 360, 360)
            lats = np.ones_like(lons) * (90. - self._A95)
            magnitudes = np.ones_like(lons)

            vecs = spherical_to_cartesian(lons, lats, magnitudes)
            rotation_matrix = construct_euler_rotation_matrix(
                0., (self.colatitude) * d2r, self.longitude * d2r)
            rotated_vecs = np.dot(rotation_matrix, vecs)
            lons, lats, magnitudes = cartesian_to_spherical(rotated_vecs.tolist())
            if south_pole is True:
                lons = lons-180.
                lats = -lats
            path = matplotlib.path.Path(np.transpose(np.array([lons, lats])))
            circ_patch = matplotlib.patches.PathPatch(
                path, transform=ccrs.Geodetic(), alpha=0.5, **kwargs)
            circ_artist = axes.add_patch(circ_patch)
            artists.append(circ_artist)
        if south_pole is False:
            artist = axes.scatter(self.longitude, self.latitude,
                                  transform=ccrs.Geodetic(), **kwargs)
        else:
            artist = axes.scatter(self.longitude-180., -self.latitude,
                                  transform=ccrs.Geodetic(), **kwargs)
        artists.append(artist)
        return artists

class PaleomagneticPole(Pole):
    """
    Subclass of Pole which represents the centroid
    of a plate. Proxy for plate position (since the
    plate is itself an extended object).
    """

    def __init__(self, longitude, latitude, age=0., sigma_age=0.0, **kwargs):

        if np.iterable(sigma_age) == 1:
            assert len(sigma_age) == 2  # upper and lower bounds
            self._age_type = 'uniform'
        else:
            self._age_type = 'gaussian'

        self._age = age
        self._sigma_age = sigma_age

        super(PaleomagneticPole, self).__init__(
            longitude, latitude, 1.0, **kwargs)

#     @property
#     def age_type(self):
#         return self._age_type

#     @property
#     def age(self):
#         return self._age

#     @property
#     def sigma_age(self):
#         return self._sigma_age

class EulerPole(Pole):
    """
    Subclass of Pole which represents an Euler pole.
    The rate is given in deg/Myr

    Here we send the rotation rate in radian/sec to the father class as the magnitude.
    """

    def __init__(self, longitude, latitude, rate, **kwargs):
        r = rate * d2r / Julian_year / 1.e6
        super(EulerPole, self).__init__(longitude, latitude, magnitude = r, **kwargs)

    @property
    def rate(self):
        # returns the angular velocity of the object that is rotating about a given Euler pole
        return self.magnitude * r2d * Julian_year * 1.e6

    def angle(self, time):
        return self.rate * time

    def speed_at_point(self, pole):
        """
        Given a point, calculate the speed that point
        rotates around the Euler pole. This assumes that
        the test pole has a radius equal to the radius of Earth,
        6371.e3 meters. It returns the speed in cm/yr.
        """
        # Give the point the radius of the earth
        point = pole._pole
#         print(point)
        point = point / np.sqrt(np.dot(point, point)) * 6371.e3
#         print(np.array([point[0], point[1], point[2]]))
        # calculate the speed
#         vel = np.cross(self._pole, np.array([point[0][0], point[1][0], point[2][0]]))
        vel = np.cross(self._pole, np.array([point[0], point[1], point[2]]))
        speed = np.sqrt(np.dot(vel, vel))

        return speed * Julian_year * 100.


class PlateCentroid(Pole):
    """
    Subclass of Pole which represents the centroid
    of a plate. Proxy for plate position (since the
    plate is itself an extended object).
    """

    def __init__(self, longitude, latitude, **kwargs):
        super(PlateCentroid, self).__init__(
            longitude, latitude, 6371.e3, **kwargs)

def two_sigma_from_kappa(kappa):
    return 140. / np.sqrt(kappa)


def kappa_from_two_sigma(two_sigma):
    return 140. * 140. / two_sigma / two_sigma
