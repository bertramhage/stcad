## Preprocessing of all 2024 data

We download data for 2024-01-01-2024-12-31, comprising 366 days. 
<!--
Result:
messages_processed:6,887,578,891
total_files_processed:366
total_filtered_messages:5,383,009,610
total_messages:6,887,578,891
total_messages_processed:6,887,578,891
total_unique_vessels:33,538
unique_vessels:33,538
-->

The data contains in total 6,887,578,891 AIS messages and after filtering based on time, LAT, LOT, SOG, COG contains 5,383,009,610 messages spanning 33,538 vessels.

Preprocessing further:
<!--
num_segments: 1314799
num_messages: 5383009610
num_discarded_filtered: 0
num_initial_voyages: 1342406
num_voyages_after_duration_filter: 726935
num_voyages_after_outlier_removal: 726935
num_outlier_removal_errors: 0
num_voyages_after_sampling: 726733
num_sampling_errors: 202
num_final_voyages: 453712
-->
After splitting segments containing pause in message of >2h we get 1,342,406 voyages and removing short voyages reduces this to 726,935 voyages. Finally we split voyages such that a voyage is max 24 hours and again remove short and low speed voyages (80% of messages have a speed < 2 knots) leading to 453,712 voyages.