README file for .csv files of the environmental conditions at PA sites (Òtest_envÓ)
These files are part of the data publication: Elith, J., Graham, C.H., Valavi, R., Abegg, M., Bruce, C., Ferrier, S., Ford, A., Guisan, A., Hijmans, R.J., Huettmann, F., Lohmann, L.G., Loiselle, B.A., Moritz, C., Overton, J.McC., Peterson, A.T., Phillips, S., Richardson, K., Williams, S., Wiser, S.K., Wohlgemuth, T. & Zimmermann, N.E. (2020) Presence-only and presence-absence data for comparing species distribution modeling methods. Biodiversity Informatics 15:69-80

These files provide group, site ids, site locations, and environmental conditions at each of the presence-absence sites in each region, and are to be used for predicting to, for later evaluation. The actual observations of presence and absence are in separate files (Òtest_paÓ), with rows (sites) matching those of these test_env files, and columns including the per-species presence-absence data. Those data are kept separate to these, to allow prediction to sites ÒblindÓ to the species observations. 

These test_env files have columns with the same names as a subset of the columns in the PO species data files found in /train_po. See the metadata files in that folder, for explanation of the meanings of each column.  

Note that for the two regions (AWT and NSW) with more than one biological group, there is a separate evaluation file for each biological group.

README file for .csv files of the species observations at PA sites (Òtest_paÓ)
These files are part of the data publication: Elith, J., Graham, C.H., Valavi, R., Abegg, M., Bruce, C., Ferrier, S., Ford, A., Guisan, A., Hijmans, R.J., Huettmann, F., Lohmann, L.G., Loiselle, B.A., Moritz, C., Overton, J.McC., Peterson, A.T., Phillips, S., Richardson, K., Williams, S., Wiser, S.K., Wohlgemuth, T. & Zimmermann, N.E. (2020) Presence-only and presence-absence data for comparing species distribution modeling methods. Biodiversity Informatics 15:69-80

These files provide group, site ids, site locations, and species observations at each of the presence-absence sites in each region, intended for evaluation of predictions made to these sites using the test_env files found in /test_env. 

The first 4 columns give the group, site id, and x and y locations - these are in identical order to their matched counterparts in test_env.
The following columns contain, per species, presence and absence observations, coded 1 and 0 respectively. 
Species codes match those in the train_po files, and explained in the report cited above.  



