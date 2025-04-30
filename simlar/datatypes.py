import numpy as np
import h5py as h5

EventType = np.dtype([
    ('run_id',    np.int32),
    ('event_id',  np.int32),
    ('input_filename', h5.string_dtype()),
    ('input_checksum', h5.string_dtype()),
])
h5_event_dtype = h5.vlen_dtype(EventType)

TPCType = np.dtype([
    ('track_id', np.int32),
    ('ix', np.int32),
    ('iy', np.int32),
    ('iz', np.int32),
    ('de', np.float32),
    ('dx', np.float32),
    ('nphotons', np.float32),
])
h5_tpc_dtype = h5.vlen_dtype(TPCType)

PMTType = np.dtype([
    ('id', np.int32),
    ('t', np.int32),
    ('nphotons', np.float32),
])
h5_pmt_dtype = h5.vlen_dtype(PMTType)
