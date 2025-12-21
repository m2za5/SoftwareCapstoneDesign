import argparse
import math
import trimesh


def normalize_mesh(input_path, output_path, target_size=1.0, rotate_x_deg=-90, rotate_y_deg=0):
    mesh = trimesh.load(input_path, process=False)
    print(f"[INFO] loaded mesh from {input_path}")
    print(f"[INFO] original bounds: {mesh.bounds}")
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    extents = mesh.extents
    max_extent = extents.max()
    if max_extent == 0:
        raise ValueError("Mesh has zero extent, check input mesh.")
    scale = target_size / max_extent
    mesh.apply_scale(scale)
    print(f"[INFO] scaled by factor {scale:.6f}")
    if rotate_x_deg != 0:
        R = trimesh.transformations.rotation_matrix(
            math.radians(rotate_x_deg), [1, 0, 0]
        )
        mesh.apply_transform(R)
        print(f"[INFO] rotated {rotate_x_deg} degrees around X-axis")
    if rotate_y_deg != 0:
        R_y = trimesh.transformations.rotation_matrix(
            math.radians(rotate_y_deg), [0, 1, 0]
        )
        mesh.apply_transform(R_y)
        print(f"[INFO] rotated {rotate_y_deg} degrees around Y-axis")

    print(f"[INFO] new bounds after normalization: {mesh.bounds}")
    mesh.export(output_path)
    print(f"[INFO] saved normalized mesh to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--size", type=float, default=1.0)
    parser.add_argument("--rotate_x", type=float, default=-90.0)
    parser.add_argument("--rotate_y", type=float, default=0.0)  # 추가된 옵션
    args = parser.parse_args()

    normalize_mesh(args.input, args.output, args.size, args.rotate_x, args.rotate_y)
