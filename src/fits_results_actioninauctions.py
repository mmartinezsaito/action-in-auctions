
from .main import ns, sp, D, B, D2
from .utils import confint_fisherinfo, bic, caic


# List of fitted models:
# model name, log-likelihood, parameters list

a = 0.285; b = 1.001 # dr naive init0(1,1,0); L = -1935.242998 
a = 0.444; b = 0.965 # dr unilatup init0(1,1,0); L = -1965.695041
a = 0.477; b = 1.003 # dr unilatup init5(1,1,5); L = -2044.705100  

fp = list()
fp.append(['11111_ffx_dr_naive_avf_11',    -4839.56, [0.025, 5.381, 5.396,  1.252,  1], [], []]) # su, minInbound, L-BFGS-B  
fp.append(['11111_ffx_dr_unilatupd_11',    -4864.24, [0.344, 1.143, 5.228,  2.026,  2.390], [], []]) # su, minInbound
fp.append(['1111_ffx_pu_naive_avf_11',     -5565.28, [       0.081, 5.472,  8.836,  5.042], [], []]) # su 
fp.append(['1111_ffx_pu_unilatupd_11',     -5638.25, [       0.089, 9.777, 12.120,  4.999], [], []]) # su
fp.append(['1111_ffx_naive_3p_nudger',     -3094.51, [       0.121, 0.009,  9.450,  6.443], [], []]) # su
fp.append(['1111_ffx_naive_gausnudger',    -2128.26, [       0.228, 0.075,  1.018,  1.134], [], []]) # su
fp.append(['111_ffx_kalman_nudger',        -2049.24, [       2.556, 6.900,  5.844,       ], [], []])
fp.append(['null',                         -6325.64, [                                   ], [], []])
fp.append(['saturated',                        0.00, ns * [0],                               [], []])
fp.append(['11_ffx_dr_gausnudger1',        -1742.26, [0.430, 1.004,                      ], [], []]) #fp.append(['111_ffx_dr_gausnudger2',-1742.26, [0.430, 1.004, 0.766], [], []])
fp.append(['11111_ffx_dr_naive_avf_101',   -5044.43, [0.280, 1.190,     1,    100,      1], [], []]) # su
fp.append(['11111_ffx_dr_unilatupd_101',   -4809.60, [0.319, 1.027, 5.334,  1.755,  4.214], [], []]) # su 

fp.append(['11_ffx_dr_unilatupd_101',      -4820.04, [0.319, 1.027], [], []])  

fp.append(['1111_ffx_pu_naive_avf_101',    -5464.32, [       0.297, 9.994, 31.973,  4.025], [], []]) # su
fp.append(['1111_ffx_pu_unilatupd_101',    -5129.35, [       0.110, 3.963,  5.301,  1.980], [], []]) # su
fp.append(['11331_ffx_dr_unilatupd_11',    -4671.41, [0.029, 2.351, 4.068, 2.791, 4.664, 3.026, 6.930, 2.205, 2.000], [], []]) # su, minInbound 
fp.append(['11331_ffx_dr_naive_avf_11',    -4566.06, [0.041, 2.326, 4.521, 2.466, 4.886, 2.784, 7.169, 2.551, 2.000], [], []]) # su, minInbound
fp.append(['1331_ffx_pu_naive_avf_11',     -5412.98, [       0.144, 4.548, 2.568, 4.904, 2.728, 6.319, 1.823, 2.653], [], []]) # fa 
fp.append(['1331_ffx_pu_unilatupd_11',     -5139.37, [       0.134, 4.848, 2.114, 4.928, 2.117, 5.373, 1.548, 3.194], [], []]) # su
#18
fp.append(['131_ffx_dr_gausnudger1',       -1741.19, [0.447, 1.010, 4.817, 5.177, 6.483                            ], [], []]) # su
fp.append(['131_ffx_kalman_nudger1',       -1882.53, [       4.366, 3.993, 4.782, 7.679, 1.572                     ], [], []])
fp.append(['11331_ffx_dr_naive_avf_101',   -4230.97, [0.125, 2.176, 4.725, 2.089, 4.466, 2.740, 6.595, 1.911, 2.118], [], []]) # su 
fp.append(['11331_ffx_dr_unilatupd_101',   -4377.43, [0.122, 1.353, 4.279, 2.699, 4.769, 2.428, 6.004, 1.567, 2.769], [], []]) # 
fp.append(['1331_ffx_pu_naive_avf_101',    -4518.00, [       0.395, 4.092, 3.596, 4.843, 3.656, 7.217, 3.180, 1.743], [], []]) # 
fp.append(['1331_ffx_pu_unilatupd_101',    -5084.51, [       0.127, 3.494, 4.518, 5.338, 2.433, 7.953, 3.848, 1.211], [], []]) # fa
#24
fp.append(['111_ffx_naive_gausnudger1',    -2231.15, [       0.134, 0.080, 1.886       ], [], []]) # su 
fp.append(['111_ffx_naive_lepkurnudger1',  -2192.85, [       0.060, 0.052, 1.704       ], [], []]) # su 
fp.append(['1111_ffx_naive_gausnudger2',   -2098.37, [       0.156, 0.049, 1.304, 1.463], [], []]) # su 
fp.append(['1111_ffx_naive_lepkurnudger2', -2128.26, [       0.228, 0.075, 1.018, 1.134], [], []]) # su 
fp.append(['11_ffx_dr_gausnudger1',        -1742.26, [0.430,               1.004       ], [], []]) # su
fp.append(['11_ffx_dr_lepkurnudger1',      -1494.84, [0.719,               0.624       ], [], []]) # su
fp.append(['111_ffx_dr_gausnudger2',       -1735.81, [0.426,               0.958, 1.128], [], []]) # su
fp.append(['111_ffx_dr_lepkurnudger2',     -1490.77, [0.719,               0.593, 0.713], [], []]) # su
fp.append(['1111_ffx_dr_lepkurnudger31',   -1511.86, [       0.100, 0.129, 0.609, 0.707], [], []]) # su
fp.append(['1111_ffx_dr_lepkurnudger32d',  -1490.77, [0.719, 4e-05,        0.593, 0.713], [], []]) # su
fp.append(['1111_ffx_dr_lepkurnudger32u',  -1481.06, [0.705,        0.070, 0.594, 0.689], [], []]) # su
fp.append(['1111_ffx_dr_lepkurnudger33',   -1482.77, [0.728, 0.105,        0.592, 0.700], [], []]) # su
fp.append(['11111_ffx_dr_lepkurnudger4',   -1470.01, [0.684, 0.057, 0.109, 0.586, 0.688], [], []]) # su
fp.append(['1111111_ffx_dr_lepkurnudger5', -1446.45, [0.700,-0.017, 0.061, 0.589, 0.628, 0.693, 0.360], [], []]) # su, overfitted?
fp.append(['11111_ffx_dr_lepkurnudger6',   -1463.53, [0.740,               0.618, 0.600, 0.631, 0.389], [], []]) # su, overfitted?
#fp.append(['33311_ffx_dr_lepkurnudger4',       0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [], []]) # su

fp.append(['11_ffx_kalman_gausnudger1',    -1993.77, [       4.110,        2.306], [], []]) # su
fp.append(['11_ffx_kalman_lepkurnudger1',  -1957.40, [       2.588,        2.789], [], []]) # su
fp.append(['111_ffx_kalman_gausnudger2',   -1870.02, [       3.338, 1.533, 3.276], [], []]) # su
fp.append(['111_ffx_kalman_lepkurnudger2', -1724.51, [       1.853, 1.554, 4.193], [], []]) # su 

fp.append(['11_ffx_dr_unilatupd_101',      -4734.98, [0.115, 1.044], [], []])

ns_s = 60
rp = []
"""using +- std"""
rp.append(['null',                     sp.log10(1/len(B)) * 60, [], [], []])
rp.append(['saturated',                                    0.0, 60 * [0], [], []])
rp.append(['rfx_neuroshyuka_dr_avf_naive_2p',   (100.65, 6.66), [(0.0158, 0.0657), (1.4413, 0.7309)], [], []]) # all isconv
rp.append(['rfx_neuroshyuka_dr_avf_naive11_2p', (101.13, 1.00), [(0.0018, 0.0010), (1.4036, 0.1106)], [], []]) # all isconv, with SEM
rp.append(['rfx_neuroshyuka_dr_avf_cntfac_2p',   (96.19, 9.06), [(0.1961, 0.2397), (1.3687, 0.6084)], [], []]) # all isconv
rp.append(['rfx_neuroshyuka_nvlepkurnudger_4p',  (35.90, 9.99), [(0.3114, 0.1872), (0.1092, 0.1081), (0.7848, 0.2743), (0.8188, 0.4504)], [], []]) # 46/47 isconv
rp.append(['rfx_neuroshyuka_drgausnudger1_2p',   (31.79,11.01), [(0.5627, 0.2547), (0.8921, 0.3785)], [], []]) # all isconv
rp.append(['rfx_neuroshyuka_drlepkurnudger7_5p', (28.04,21.03), [(0.5845, 0.3066), (0.6114, 0.3039), (0.7888, 0.3091), (0.5567, 0.3382), (0.3022, 0.1440)], [], []]) #17/47 conv

fp29 = []
fp29.append(['dr_avf_agent101', -2999.91436802, [0.0, 1.04181586388], {'yterushkina': -100.66607462273076, 'erydkina': -101.49590081372197, 'emikhedova': -114.68805925851763, 'opetrova': -97.129668126925694, 'ezaikina': -96.743987955188487, 'kisaeva': -102.70087250867412, 'pkikot': -120.76524406506984, 'opolshina': -97.894599395298215, 'dkulik': -98.264053908377932, 'myurevich': -103.94952237720013, 'jsaakyan': -137.06603688225013, 'kakopyan': -109.09736934881276, 'mtodua': -104.34296213311019, 'mbaranova': -97.344722771121468, 'ndagaev': -99.60549569901346, 'dpozdeeva': -99.89506785453311, 'ozabrodskaya': -97.148954580891427, 'elevchenko': -100.41164048825263, 'akonik': -109.5743000172332, 'ekucherova': -100.39770194471338, 'rvakhrushev': -103.81153271544113, 'eokorokova': -95.55306663456696, 'vbudykin': -98.370229333174237, 'azalyaev': -102.2527032883126, 'ekarelova': -93.749535693665763, 'msoloreva': -97.248714043946663, 'rryzhikh': -105.10949798265615, 'mivanov': -113.39640232649857, 'taleshkovskaya': -101.24045124655704}])
fp29.append(['dr_avf_agent11', -3012.66217231, [0.0, 0.990300180556], {'rvakhrushev': -105.42662239055143, 'emikhedova': -115.79479590630719, 'ozabrodskaya': -97.770226997961956, 'taleshkovskaya': -101.34451052895716, 'mivanov': -115.19739796122938, 'ekarelova': -95.292745164786112, 'mtodua': -104.44930610148727, 'pkikot': -117.49331815383908, 'rryzhikh': -105.59585684612036, 'kakopyan': -109.38771529606426, 'azalyaev': -102.23161326804549, 'vbudykin': -98.441211352953133, 'mbaranova': -97.638839861205412, 'ekucherova': -101.29583708084431, 'opolshina': -99.165604462888226, 'ezaikina': -97.356410025757128, 'msoloreva': -98.143499125296103, 'dpozdeeva': -101.12128316909437, 'akonik': -110.49563736566358, 'opetrova': -97.54438239371612, 'eokorokova': -96.168701203175203, 'ndagaev': -99.852736833361092, 'kisaeva': -102.49613717930646, 'yterushkina': -101.09036874263865, 'elevchenko': -101.27975362788598, 'myurevich': -104.82877363003381, 'jsaakyan': -134.70241551608927, 'erydkina': -102.0989009845718, 'dkulik': -98.957571140283719}])
fp29.append(['dr_avfcntfac_agent101', -2957.47211189, [0.0878493795975, 1.00181644388], {'yterushkina': -108.6058392656219, 'erydkina': -94.751597377625032, 'emikhedova': -111.41664373110736, 'opetrova': -94.992034340003556, 'ezaikina': -97.944773240067789, 'kisaeva': -104.62814238652555, 'pkikot': -131.41477206688677, 'opolshina': -101.90905390996582, 'dkulik': -97.318372705400549, 'myurevich': -100.68153966889794, 'jsaakyan': -139.79082747721495, 'kakopyan': -105.48473942493014, 'mtodua': -103.42833092567977, 'mbaranova': -94.342498423375872, 'ndagaev': -93.002937340631277, 'dpozdeeva': -92.232226434994914, 'ozabrodskaya': -94.547816108480859, 'elevchenko': -95.818715520030437, 'akonik': -112.08236569951683, 'ekucherova': -94.535181445549611, 'rvakhrushev': -100.89259941856373, 'eokorokova': -90.499093861955572, 'vbudykin': -96.619904633681799, 'azalyaev': -98.622180327131943, 'ekarelova': -96.954808769587942, 'msoloreva': -92.847775605357072, 'rryzhikh': -101.07797940156708, 'mivanov': -107.29258178919221, 'taleshkovskaya': -103.73678059162324}])
fp29.append(['naive_lepkurnudger2', -1341.61756973, [0.199676544805, 0.0644017193221, 1.0626618341, 1.16241818677], {'jsaakyan': -77.299801405254769, 'yterushkina': -53.43182380075983, 'ozabrodskaya': -35.700257961266594, 'erydkina': -38.076574209473236, 'emikhedova': -51.221019367839361, 'mivanov': -57.263076052803306, 'opetrova': -35.819717436644005, 'opolshina': -49.690772383795341, 'rryzhikh': -49.190526148996469, 'azalyaev': -41.107212026897393, 'myurevich': -54.765775148322668, 'ekucherova': -35.855935586119408, 'pkikot': -80.426150567945683, 'eokorokova': -35.49886329201901, 'ekarelova': -40.780902636376709, 'ezaikina': -44.045306348061537, 'vbudykin': -35.71529565928703, 'kisaeva': -49.602554644383048, 'msoloreva': -32.962040276216719, 'elevchenko': -36.959703156953189, 'kakopyan': -52.3259773854275, 'ndagaev': -40.635878082951301, 'dkulik': -44.019868410615025, 'akonik': -51.680736941653052, 'mbaranova': -38.490033471797801, 'mtodua': -47.338321086896194, 'dpozdeeva': -41.479781115632505, 'taleshkovskaya': -46.719087897351734, 'rvakhrushev': -43.514577230371678}])
fp29.append(['dr_gausnudger1', -1137.20942315, [0.381258409263, 1.0907059209], {'jsaakyan': -98.20697790528321, 'yterushkina': -45.566665981057213, 'ozabrodskaya': -28.811648622470731, 'erydkina': -30.246019016007551, 'emikhedova': -47.930523515360193, 'mivanov': -43.982027477699823, 'opetrova': -35.249592190937868, 'opolshina': -38.408254652434486, 'rryzhikh': -40.533479701139854, 'azalyaev': -36.29580968692833, 'myurevich': -32.073334776376157, 'ekucherova': -34.556359842614434, 'pkikot': -62.794979085101147, 'eokorokova': -29.336815108302847, 'ekarelova': -29.851314913046689, 'ezaikina': -35.781162480141617, 'vbudykin': -31.646790053706944, 'kisaeva': -50.206260303707005, 'msoloreva': -33.688016620186374, 'elevchenko': -31.594738408127078, 'kakopyan': -32.198839430016925, 'ndagaev': -37.150560049175326, 'dkulik': -35.542230987596604, 'akonik': -33.909677261142939, 'mbaranova': -29.269498798741811, 'mtodua': -45.659568703832761, 'dpozdeeva': -29.673432562993831, 'taleshkovskaya': -40.386275732603806, 'rvakhrushev': -36.658569278717245}])
fp29.append(['dr_lepkurnudger7', -990.245414987, [0.531029005072, 0.702609061939, 0.79252342973, 0.653085584372, 0.387249139955], {'jsaakyan': -79.098095811203549, 'yterushkina': -48.048196134412066, 'ozabrodskaya': -19.522613697381257, 'erydkina': -15.758937947111379, 'emikhedova': -51.337841422008019, 'mivanov': -32.570688862375462, 'opetrova': -32.461566141490721, 'opolshina': -41.818329171129335, 'rryzhikh': -31.179891823831852, 'azalyaev': -35.358572095166373, 'myurevich': -24.999388086240952, 'ekucherova': -30.028800238495382, 'pkikot': -56.301761215328376, 'eokorokova': -16.430269793316779, 'ekarelova': -24.462045606757311, 'ezaikina': -32.052257682183843, 'vbudykin': -25.552413434727189, 'kisaeva': -52.968580355309136, 'msoloreva': -28.19926783438228, 'elevchenko': -20.019345727645803, 'kakopyan': -26.940189409527576, 'ndagaev': -26.208438769358164, 'dkulik': -32.586903689922075, 'akonik': -35.239416262070002, 'mbaranova': -20.967527104500718, 'mtodua': -51.137204101933662, 'dpozdeeva': -20.654568715826752, 'taleshkovskaya': -44.294505655527232, 'rvakhrushev': -34.047798198264204}])
fp29_nlls = [sp.mean(list(fp29[i][3].values())) for i in range(6)], [sp.stats.sem(list(fp29[i][3].values())) for i in range(6)]





