#    def segmentation_generator(
#         self,
#         patient_generator: PatientGenerator,
#         samples_per_patient: int | list[int] = 1,
#     ) -> SampleGenerator:
#         """Generate frames and segment labels.

#         Args:
#             patient_generator (PatientGenerator): Patient Generator
#             samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.
#         Returns:
#             SampleGenerator: Sample generator
#         Yields:
#             Iterator[SampleGenerator]
#         """

#         for _, pt in patient_generator:
#             # NOTE: [:] will load all data into RAM- ideal for small dataset
#             data = pt["data"][:]
#             segs = pt["segmentations"][:]
#             fids = pt["fiducials"][:]

#             if self.sampling_rate != self.target_rate:
#                 ratio = self.target_rate / self.sampling_rate
#                 data = pk.signal.resample_signal(data, self.sampling_rate, self.target_rate, axis=0)
#                 segs[:, (SEG_BEG_IDX, SEG_END_IDX)] = segs[:, (SEG_BEG_IDX, SEG_END_IDX)] * ratio
#                 fids[:, FID_LOC_IDX] = fids[:, FID_LOC_IDX] * ratio
#             # END IF

#             # Create segmentation mask
#             labels = np.zeros_like(data)
#             for seg_idx in range(segs.shape[0]):
#                 seg = segs[seg_idx]
#                 labels[seg[SEG_BEG_IDX] : seg[SEG_END_IDX], seg[SEG_LEAD_IDX]] = seg[SEG_LBL_IDX]
#             # END FOR

#             start_offset = max(0, segs[0][SEG_BEG_IDX] - 100)
#             stop_offset = max(0, data.shape[0] - segs[-1][SEG_END_IDX] + 100)
#             for _ in range(samples_per_patient):
#                 # Randomly pick an ECG lead
#                 lead = np.random.randint(data.shape[1])
#                 # Randomly select frame within the segment
#                 frame_start = np.random.randint(start_offset, data.shape[0] - self.frame_size - stop_offset)
#                 frame_end = frame_start + self.frame_size
#                 x = data[frame_start:frame_end, lead].astype(np.float32).reshape((self.frame_size,))
#                 y = labels[frame_start:frame_end, lead].astype(np.int32)
#                 yield x, y
#             # END FOR
#         # END FOR
