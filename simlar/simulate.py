from .datatypes import TPCType, PMTType, EventType, h5_tpc_dtype, h5_pmt_dtype, h5_event_dtype
from .models import PhotonTransport
from .utils import load_config, get_device, get_file_checksum
import json, hashlib, torch, time
import h5py as h5
import numpy as np 

INVALID_TRACK_ID=pow(2,31)-1

class DetectorSimulator:

    def __init__(self,cfg):
        '''
        Initialize the DetectorSimulator with a configuration file.
        Parameters
        ----------
        cfg : str or dict
            The configuration file path or a dictionary containing the configuration parameters.
        '''
        self.configure(cfg)

    def configure(self,cfg):
        '''
        Configure the DetectorSimulator with a configuration file.
        Parameters
        ----------
        cfg : str or dict
            The configuration file path or a dictionary containing the configuration parameters.
        '''
        if type(cfg) is str:
            cfg = load_config(cfg)

        assert type(cfg) is dict, 'cfg should be a dictionary'

        self.input_file  = cfg['SIMULATION']['IO']['input_file']
        self.output_file = cfg['SIMULATION']['IO']['output_file']
        self.key_pstep = cfg['SIMULATION']['IO']['input_key_pstep']
        self.key_part  = cfg['SIMULATION']['IO']['input_key_particle']
        self.key_event = cfg['SIMULATION']['IO']['input_key_event']
        self.num_entries = cfg['SIMULATION']['IO']['num_entries']
        self.num_skip    = cfg['SIMULATION']['IO']['num_skip']
        self.sim_pmt = PhotonTransport(cfg)
        self.volume = torch.as_tensor([cfg['GEOMETRY']['TPC']['active_volume']['x'], 
        cfg['GEOMETRY']['TPC']['active_volume']['y'],
        cfg['GEOMETRY']['TPC']['active_volume']['z']]).to(torch.float32)
        self.voxel_size = torch.as_tensor(cfg['SIMULATION']['TRUTH']['voxel_size']).to(torch.float32)

        self.config = dict(cfg)
        self.to(get_device(cfg.get('DEVICE','cpu')))

    def to(self,device):
        '''
        Move the model and all its torch.Tensor attributes to a specified device (CPU or GPU).

        Parameters
        ----------
        device : str
            The device to move the model to, e.g., 'cpu' or 'cuda'.
        '''
        self.device = device
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))

        self.sim_pmt.to(device)
        self.config['DEVICE'] = str(device)

    def process(self):
        '''
        Process the input file and generate the output file with simulated data.
        '''

        print('\nStart event processing')
        with h5.File(self.output_file,'w') as fout:

            print('Opening output file :',self.output_file)            
            fout.attrs['config'] = json.dumps(self.config)
            fout.attrs['pmt_positions'] = self.sim_pmt.pmt_positions.cpu().detach().numpy()
            fout.attrs['pmt_ids'] = self.sim_pmt.pmt_ids.cpu().detach().numpy()

            input_checksum = get_file_checksum(self.input_file)

            with h5.File(self.input_file,'r') as fin:
                print('Opening input file  :',self.input_file)
                ds_step_v  = fin[self.key_pstep]
                ds_part_v  = fin[self.key_part]
                ds_event_v = fin[self.key_event]
            
                num_entries = len(ds_step_v)
                print('Input file contains',num_entries,'entries')
                if self.num_skip >= num_entries:
                    print(f"Skipping all entries in {self.input_file}")
                    return

                if self.num_skip <= 0:
                    self.num_skip = 0
                else:
                    print(f"Skipping {self.num_skip} entries")

                if self.num_entries < 0:
                    self.num_entries = num_entries
                if num_entries < (self.num_skip + self.num_entries):
                    print('ERROR: Requested to skip',self.num_skip,'and process',self.num_entries)
                    print('       But the input file contains only',num_entries)
                    return

                num_entries = self.num_entries

                void_entries = (ds_event_v['num_particles'] < 1) | (ds_event_v['num_steps'] < 1)
                void_entries = void_entries[self.num_skip:self.num_skip+self.num_entries]

                ds_id  = fout.create_dataset('event', (num_entries-void_entries.sum(),), dtype=h5_event_dtype)
                ds_tpc = fout.create_dataset('tpc', (num_entries-void_entries.sum(),), dtype=h5_tpc_dtype)
                ds_pmt = fout.create_dataset('pmt', (num_entries-void_entries.sum(),), dtype=h5_pmt_dtype)
                ds_part_out = fout.create_dataset('particle/geant4', (num_entries-void_entries.sum(),), dtype=ds_part_v.dtype)
                print(f"Processing {num_entries} entries from {self.input_file}...")
                output_index = 0
                for index in range(num_entries):
                    entry = index + self.num_skip
                    if void_entries[index]:
                        # Skip empty entries
                        print(f"Skipping entry {entry} with no particles or steps")
                        continue
                    ds_event = ds_event_v[entry]
                    ds_step  = ds_step_v[entry]
                    ds_part  = ds_part_v[entry]
                    
                    # Exclude segments w/ invlaid track ID 
                    mask = ~(ds_step['track_id'] == INVALID_TRACK_ID)
                    ds_step = ds_step[mask]
                    #
                    # calculate num photons
                    #
                    mass = ds_part['mass'][ds_step['track_id']]
                    # Exclude segments with zero mass (photons)
                    mask = mass > 0
                    ds_step = ds_step[mask]
                    mass = mass[mask]

                    t0=time.time()
                    
                    betagamma = torch.as_tensor(ds_step['p'] / mass).to(self.device)
                    de = torch.as_tensor(ds_step['de']).to(self.device)
                    dx = torch.as_tensor(ds_step['dx']).to(self.device)
                    
                    photons = self.sim_pmt.de_to_photons(de,betagamma)
                    
                    # convert position into voxels
                    pos = torch.as_tensor(np.column_stack([ds_step['x'],ds_step['y'],ds_step['z'],ds_step['t']])).to(self.device)
                    
                    idx  = torch.zeros(size=(len(pos),4),dtype=torch.int32)
                    for i in range(3):
                        idx[:,i] = ((pos[:,i] - self.volume[i][0]) / self.voxel_size[i] - 0.5).to(torch.int32)
                    idx[:,3] = torch.as_tensor(ds_step['track_id'])
                    
                    res0, idx_map = torch.unique(idx, dim=0, return_inverse=True)
                    idx_map = idx_map.to(self.device)
                    res1 = torch.zeros(size=(3,len(res0)),dtype=torch.float32).to(self.device)
                    res1[0,:].index_add_(0, idx_map, de )
                    res1[1,:].index_add_(0, idx_map, dx )
                    res1[2,:].index_add_(0, idx_map, photons)

                    res0 = res0.cpu().detach().numpy()
                    res1 = res1.cpu().detach().numpy()
                    tpc = np.zeros(shape=(len(res0)),dtype=TPCType)
                    tpc['track_id'] = res0[:,3]
                    tpc['ix'] = res0[:,0]
                    tpc['iy'] = res0[:,1]
                    tpc['iz'] = res0[:,2]
                    tpc['de'] = res1[0,:]
                    tpc['dx'] = res1[1,:]
                    tpc['nphotons'] = res1[2,:]
                                    
                    pmt_ids, pmt_ts, pmt_nphotons = self.sim_pmt.get_pe(photons, pos)

                    pmt = np.zeros(shape=(len(pmt_ids)),dtype=PMTType)
                    pmt['id'] = pmt_ids.cpu().detach().numpy()
                    pmt['t' ] = pmt_ts.cpu().detach().numpy()
                    pmt['nphotons'] = pmt_nphotons.cpu().detach().numpy()

                    event = np.zeros(shape=(len(ds_event)),dtype=EventType)
                    event['run_id'] = ds_event['run_id']
                    event['event_id'] = ds_event['event_id']
                    event['input_filename'][:] = self.input_file
                    event['input_checksum'][:] = input_checksum

                    part_out = np.zeros(shape=(len(ds_part),),dtype=ds_part_v.dtype)
                    part_out[:] = ds_part

                    ds_tpc[output_index]=tpc
                    ds_pmt[output_index]=pmt
                    ds_id[output_index]=event
                    ds_part_out[output_index]=part_out
                    output_index += 1

                    print(f'Entry {entry:4d} with {len(ds_step):6d} segments: {time.time()-t0:.3f} seconds')

                assert output_index == len(ds_tpc), f"Output index {output_index} does not match the number of entries {len(ds_tpc)}"
    