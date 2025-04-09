import torch

def generate_pmt_positions(lx, ly, lz, spacing_y, spacing_z, gap_pmt_active):
    '''
    Generate hexagonal grid of PMTs on the y-z plane of a cubic detector
    :param lx: length of the cube in x direction
    :param ly: length of the cube in y direction
    :param lz: length of the cube in z direction
    :param spacing_y: spacing between PMTs in y direction
    :param spacing_z: spacing between PMTs in z direction
    :param gap_pmt_anode: spacing between a pmt plane and the closest active volume boundary
    :return:
    pmt_coords: coordinates of the PMTs
    '''

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


