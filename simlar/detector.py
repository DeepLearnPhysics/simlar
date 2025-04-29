import torch

def generate_pmt_positions(lx, ly, lz, spacing_y, spacing_z, gap_pmt_active):
    """
    Generate PMT positions in a hexagonal grid pattern.
    Parameters
    ----------
    lx : float
        Length of the active volume in the x-direction.
    ly : float
        Length of the active volume in the y-direction.
    lz : float
        Length of the active volume in the z-direction.
    spacing_y : float
        Spacing between PMTs in the y-direction.
    spacing_z : float
        Spacing between PMTs in the z-direction.
    gap_pmt_active : float
        Gap between the PMT and the active volume.
    Returns
    -------
    pmt_coords : torch.Tensor
        Tensor containing the x, y, z coordinates of the PMTs.
    pmt_ids : torch.Tensor
        Tensor containing the IDs of the PMTs.
    """

    grid_y = int(ly / spacing_y)
    grid_z = int(lz / spacing_z)
    print(f"Total PMT number is {2 * grid_y * grid_z}")

    # Generate hexagonal grid coordinates
    y_side, z_side = torch.meshgrid(torch.arange(grid_y), torch.arange(grid_z))
    y_side = y_side * spacing_y - ly / 2
    z_side = z_side * spacing_z - lz / 2
    y_side = y_side.to(torch.float32)
    z_side = z_side.to(torch.float32)
    for i in range(y_side.shape[1]):
        if i % 2 == 1:
            y_side[:, i] += spacing_y / 2

    y = torch.tile(y_side, (2, 1, 1)).flatten()#reshape(2, -1)
    z = torch.tile(z_side, (2, 1, 1)).flatten()#reshape(2, -1)

    num_lo_pmt = num_hi_pmt = int(len(y)/2)
    lo_x_value = -lx/2-gap_pmt_active
    hi_x_value = lx/2+gap_pmt_active
    x = torch.concat((torch.full((num_lo_pmt,), lo_x_value),torch.full((num_hi_pmt,), hi_x_value)))
    # print(x, y, z)
    # Shift every other row to create a hexagonal pattern

    #pmt_coords = torch.column_stack((x.flatten(), y.flatten(), z.flatten()))
    # change to swap between the sides
    pmt_coords = torch.stack((x,y,z), dim=-1)
    return pmt_coords, torch.arange( pmt_coords.shape[0], dtype=torch.int32)


def pmt_collection_efficiency(x: torch.Tensor, **kwargs):
    '''
    Calculates the sensor angular efficiency from a photon
    Parameters
    ----------
    x: tensor of angles, shape (N_photon, N_pmt)
    Returns
    -------
    efficiency: tensor of efficiencies (N_photon, N_pmt)
    '''
    sigmoid_coeff = kwargs.get('sigmoid_coeff', 0.15)
    efficiency = 1 / (1 + torch.exp(-sigmoid_coeff * 180. / torch.pi * (torch.abs(torch.pi / 2 - x) - torch.pi / 4)))
    return efficiency
    