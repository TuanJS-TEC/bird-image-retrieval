import os

import matplotlib.pyplot as plt
import pandas as pd

from .common import _debug_log


def find_perching_attribute_ids(attr_names_df: pd.DataFrame) -> list[int]:
    print("\n[INFO] Kiem tra attributes lien quan den tu the...")
    keywords = r"perch|perching|sitting|fly|flying|flight|upright"
    matched = attr_names_df[attr_names_df["attr_name"].str.contains(keywords, case=False, na=False)]
    perch_like = matched["attr_name"].str.contains(r"perch|perching|sitting|upright", case=False, na=False)
    fly_like = matched["attr_name"].str.contains(r"fly|flying|flight", case=False, na=False)
    _debug_log(
        run_id="debug-1",
        hypothesis_id="H1",
        location="cub_pipeline/filtering.py",
        message="Attribute keyword matching summary",
        data={
            "matched_total": int(len(matched)),
            "perch_like_count": int(perch_like.sum()),
            "fly_like_count": int(fly_like.sum()),
            "sample_matched": matched["attr_name"].head(12).tolist(),
        },
    )
    if len(matched) > 0:
        print("  Tim thay attributes lien quan:")
        print(matched.to_string(index=False))
        return matched["attr_id"].astype(int).tolist()
    print("  [!] Khong tim thay attribute tu the ro rang.")
    return []


def compute_perching_score(
    master_df: pd.DataFrame,
    attr_labels_df: pd.DataFrame,
    perching_attr_ids: list[int],
    part_locs_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("BUOC 2: Loc anh chim dang dau (khong bay)")
    print("=" * 60)
    out = master_df.copy()
    out["bbox_ratio"] = out["width"] / out["height"]
    out["bbox_area"] = out["width"] * out["height"]
    out["leg_visible_any"] = False
    out["wing_visible_any"] = False
    out["left_eye_visible"] = False
    out["right_eye_visible"] = False
    out["beak_visible"] = False
    out["tail_visible"] = False
    out["beak_tail_dx_norm"] = -1.0
    out["beak_tail_dy_norm"] = -1.0
    out["leg_y_norm_max"] = -1.0
    out["wing_span_norm"] = -1.0

    has_attr_signal = False
    pose_attr_subset = pd.DataFrame(columns=["img_id", "attr_id", "certainty_id"])
    if perching_attr_ids:
        subset = attr_labels_df[attr_labels_df["attr_id"].isin(perching_attr_ids)].copy()
        subset = subset[(subset["is_present"] == 1) & (subset["certainty_id"] >= 3)]
        pose_attr_subset = subset.copy()
        perch_img_ids = set(subset["img_id"].astype(int).tolist())
        out["attr_perching"] = out["img_id"].isin(perch_img_ids)
        has_attr_signal = len(perch_img_ids) > 0
    else:
        out["attr_perching"] = False

    out["bbox_perching"] = out["bbox_ratio"] <= 2.0
    if has_attr_signal:
        out["likely_perching"] = out["attr_perching"]
        decision_policy = "attr_only_when_available"
    else:
        out["likely_perching"] = out["bbox_perching"]
        decision_policy = "bbox_fallback"

    if part_locs_df is not None and len(part_locs_df) > 0:
        legs = part_locs_df[(part_locs_df["part_id"].isin([8, 12])) & (part_locs_df["visible"] == 1)]
        wings = part_locs_df[(part_locs_df["part_id"].isin([9, 13])) & (part_locs_df["visible"] == 1)]
        left_eyes = part_locs_df[(part_locs_df["part_id"] == 7) & (part_locs_df["visible"] == 1)]
        right_eyes = part_locs_df[(part_locs_df["part_id"] == 11) & (part_locs_df["visible"] == 1)]
        beaks = part_locs_df[(part_locs_df["part_id"] == 2) & (part_locs_df["visible"] == 1)]
        tails = part_locs_df[(part_locs_df["part_id"] == 14) & (part_locs_df["visible"] == 1)]
        out["leg_visible_any"] = out["img_id"].isin(set(legs["img_id"].astype(int).tolist()))
        out["wing_visible_any"] = out["img_id"].isin(set(wings["img_id"].astype(int).tolist()))
        out["left_eye_visible"] = out["img_id"].isin(set(left_eyes["img_id"].astype(int).tolist()))
        out["right_eye_visible"] = out["img_id"].isin(set(right_eyes["img_id"].astype(int).tolist()))
        out["beak_visible"] = out["img_id"].isin(set(beaks["img_id"].astype(int).tolist()))
        out["tail_visible"] = out["img_id"].isin(set(tails["img_id"].astype(int).tolist()))

        beak_points = {int(r["img_id"]): (float(r["x"]), float(r["y"])) for _, r in beaks.iterrows()}
        tail_points = {int(r["img_id"]): (float(r["x"]), float(r["y"])) for _, r in tails.iterrows()}
        left_leg_points = {int(r["img_id"]): (float(r["x"]), float(r["y"])) for _, r in legs[legs["part_id"] == 8].iterrows()}
        right_leg_points = {int(r["img_id"]): (float(r["x"]), float(r["y"])) for _, r in legs[legs["part_id"] == 12].iterrows()}
        left_wing_points = {int(r["img_id"]): (float(r["x"]), float(r["y"])) for _, r in wings[wings["part_id"] == 9].iterrows()}
        right_wing_points = {int(r["img_id"]): (float(r["x"]), float(r["y"])) for _, r in wings[wings["part_id"] == 13].iterrows()}

        dx_norm_values = []
        dy_norm_values = []
        leg_y_norm_values = []
        wing_span_norm_values = []
        for _, rr in out.iterrows():
            img_id = int(rr["img_id"])
            if img_id in beak_points and img_id in tail_points and float(rr["width"]) > 0 and float(rr["height"]) > 0:
                bx, by = beak_points[img_id]
                tx, ty = tail_points[img_id]
                dx_norm_values.append(abs(bx - tx) / float(rr["width"]))
                dy_norm_values.append(abs(by - ty) / float(rr["height"]))
            else:
                dx_norm_values.append(-1.0)
                dy_norm_values.append(-1.0)
            if float(rr["height"]) > 0:
                leg_y_candidates = []
                if img_id in left_leg_points:
                    leg_y_candidates.append((left_leg_points[img_id][1] - float(rr["y"])) / float(rr["height"]))
                if img_id in right_leg_points:
                    leg_y_candidates.append((right_leg_points[img_id][1] - float(rr["y"])) / float(rr["height"]))
                leg_y_norm_values.append(float(max(leg_y_candidates)) if leg_y_candidates else -1.0)
            else:
                leg_y_norm_values.append(-1.0)
            if img_id in left_wing_points and img_id in right_wing_points and float(rr["width"]) > 0:
                wx_l = left_wing_points[img_id][0]
                wx_r = right_wing_points[img_id][0]
                wing_span_norm_values.append(float(abs(wx_r - wx_l) / float(rr["width"])))
            else:
                wing_span_norm_values.append(-1.0)
        out["beak_tail_dx_norm"] = dx_norm_values
        out["beak_tail_dy_norm"] = dy_norm_values
        out["leg_y_norm_max"] = leg_y_norm_values
        out["wing_span_norm"] = wing_span_norm_values

        if has_attr_signal:
            one_eye_visible = out["left_eye_visible"] ^ out["right_eye_visible"]
            out["likely_perching"] = (
                out["attr_perching"]
                & out["leg_visible_any"]
                & one_eye_visible
                & out["beak_visible"]
                & out["tail_visible"]
                & (out["bbox_ratio"] <= 1.7)
                & (out["beak_tail_dx_norm"] >= 0.55)
                & (out["beak_tail_dy_norm"] >= 0.26)
                & (out["beak_tail_dy_norm"] <= 0.45)
                & (out["leg_y_norm_max"] >= 0.62)
            )
            decision_policy = "attr_relaxed_rules"

    hard_negative_output_filenames = {
        "00002_Black_Footed_Albatross_0009_34.jpg",
        "00041_Black_Footed_Albatross_0077_796114.jpg",
        "00009_Black_Footed_Albatross_0010_796097.jpg",
        "00165_Sooty_Albatross_0007_796372.jpg",
        "00144_Sooty_Albatross_0005_796342.jpg",
        "00114_Laysan_Albatross_0059_488.jpg",
        "00044_Black_Footed_Albatross_0037_796120.jpg",
        "00163_Sooty_Albatross_0065_796367.jpg",
        "00172_Sooty_Albatross_0079_796389.jpg",
        "00049_Black_Footed_Albatross_0036_796127.jpg",
        "00140_Sooty_Albatross_0045_1162.jpg",
        "00067_Laysan_Albatross_0071_792.jpg",
        "00971_Spotted_Catbird_0040_796820.jpg",
        "01024_Gray_Catbird_0107_20513.jpg",
        "00610_Yellow_Headed_Blackbird_0065_8481.jpg",
        "01029_Gray_Catbird_0015_21230.jpg",
        "01041_Gray_Catbird_0105_20864.jpg",
        "01043_Gray_Catbird_0002_21395.jpg",
        "01074_Yellow_Breasted_Chat_0044_22106.jpg",
        "01079_Yellow_Breasted_Chat_0058_21864.jpg",
        "01078_Yellow_Breasted_Chat_0068_21860.jpg",
        "01088_Yellow_Breasted_Chat_0100_21913.jpg",
        "01091_Yellow_Breasted_Chat_0103_21670.jpg",
        "01098_Yellow_Breasted_Chat_0094_21693.jpg",
        "01157_Eastern_Towhee_0090_22273.jpg",
        "01153_Eastern_Towhee_0112_22231.jpg",
        "00915_Cardinal_0023_19026.jpg",
        "00447_Brewer_Blackbird_0137_2680.jpg",
        "00913_Cardinal_0093_17676.jpg",
        "01067_Yellow_Breasted_Chat_0032_21823.jpg",
        "01156_Eastern_Towhee_0050_22257.jpg",
        "01155_Eastern_Towhee_0073_22247.jpg",
        "01158_Eastern_Towhee_0015_22275.jpg",
        "00219_Groove_Billed_Ani_0059_1480.jpg",
        "00259_Crested_Auklet_0074_794949.jpg",
        "00657_Yellow_Headed_Blackbird_0073_8442.jpg",
        "00270_Crested_Auklet_0076_785252.jpg",
        "00280_Crested_Auklet_0030_794937.jpg",
        "00264_Crested_Auklet_0033_794964.jpg",
        "00284_Least_Auklet_0050_1924.jpg",
        "00268_Crested_Auklet_0067_785249.jpg",
        "00459_Brewer_Blackbird_0049_2258.jpg",
        "00494_Red_Winged_Blackbird_0042_3635.jpg",
        "00491_Red_Winged_Blackbird_0005_5636.jpg",
        "00502_Red_Winged_Blackbird_0061_4196.jpg",
        "00522_Red_Winged_Blackbird_0045_4526.jpg",
        "00517_Red_Winged_Blackbird_0027_4123.jpg",
        "00606_Yellow_Headed_Blackbird_0031_8456.jpg",
        "00637_Yellow_Headed_Blackbird_0080_8601.jpg",
        "01169_Eastern_Towhee_0129_22358.jpg",
        "00885_Painted_Bunting_0078_16565.jpg",
        "00797_Lazuli_Bunting_0066_14914.jpg",
        "00632_Yellow_Headed_Blackbird_0042_8574.jpg",
        "00921_Cardinal_0105_19045.jpg",
        "00942_Cardinal_0016_17862.jpg",
        "00790_Lazuli_Bunting_0080_14893.jpg",
        "01066_Yellow_Breasted_Chat_0011_21820.jpg",
        "01065_Yellow_Breasted_Chat_0089_21804.jpg",
        "01201_Chuck_Will_Widow_0055_796973.jpg",
        "01260_Brandt_Cormorant_0022_23157.jpg",
        "01298_Red_Faced_Cormorant_0002_796275.jpg",
        "01816_Yellow_Billed_Cuckoo_0023_26637.jpg",
        "01313_Red_Faced_Cormorant_0054_796301.jpg",
        "01610_American_Crow_0080_25220.jpg",
        "01832_Yellow_Billed_Cuckoo_0061_26692.jpg",
        "01840_Yellow_Billed_Cuckoo_0074_26466.jpg",
        "01159_Eastern_Towhee_0022_22279.jpg",
        "01952_Purple_Finch_0102_27308.jpg",
        "01283_Brandt_Cormorant_0073_23259.jpg",
        "01940_Purple_Finch_0031_28175.jpg",
        "01492_Shiny_Cowbird_0081_796833.jpg",
        "04276_Florida_Jay_0081_64859.jpg",
        "04008_Rufous_Hummingbird_0045_59533.jpg",
        "04481_Tropical_Kingbird_0024_69582.jpg",
        "05145_Western_Meadowlark_0052_77781.jpg",
        "05492_White_Breasted_Nuthatch_0131_86416.jpg",
        "04593_Belted_Kingfisher_0058_70848.jpg",
        "05511_Baltimore_Oriole_0020_87066.jpg",
        "07237_Le_Conte_Sparrow_0025_795188.jpg",
        "05494_White_Breasted_Nuthatch_0070_85983.jpg",
        "07332_Nelson_Sharp_Tailed_Sparrow_0079_796934.jpg",
        "03738_Slaty_Backed_Gull_0043_796009.jpg",
        "07380_Nelson_Sharp_Tailed_Sparrow_0062_796919.jpg",
        "07454_Seaside_Sparrow_0043_796510.jpg",
        "07442_Savannah_Sparrow_0114_119750.jpg",
        "01646_Fish_Crow_0067_26124.jpg",
        "03638_Ivory_Gull_0004_49019.jpg",
        "03467_Glaucous_Winged_Gull_0086_44268.jpg",
        "02316_Scissor_Tailed_Flycatcher_0121_41843.jpg",
        "02389_Vermilion_Flycatcher_0017_42407.jpg",
        "03641_Ivory_Gull_0101_49790.jpg",
        "02821_Boat_Tailed_Grackle_0097_33759.jpg",
        "06527_Great_Grey_Shrike_0015_797031.jpg",
        "05883_White_Pelican_0079_97380.jpg",
        "07278_Lincoln_Sparrow_0036_117280.jpg",
        "08247_Artic_Tern_0065_141472.jpg",
        "07747_White_Throated_Sparrow_0078_129041.jpg",
        "07514_Song_Sparrow_0089_120894.jpg",
        "05642_Orchard_Oriole_0032_91201.jpg",
        "06024_Sayornis_0025_98620.jpg",
        "07384_Nelson_Sharp_Tailed_Sparrow_0035_796924.jpg",
        "08393_Caspian_Tern_0067_145107.jpg",
        "08297_Black_Tern_0089_144174.jpg",
        "08329_Black_Tern_0083_144083.jpg",
        "03734_Slaty_Backed_Gull_0030_796003.jpg",
        "08053_Tree_Swallow_0030_134942.jpg",
        "11451_Carolina_Wren_0095_186561.jpg",
        "09315_Black_And_White_Warbler_0077_160440.jpg",
        "11453_Carolina_Wren_0032_186566.jpg",
        "11576_Marsh_Wren_0004_188188.jpg",
        "11493_House_Wren_0087_187946.jpg",
        "11679_Winter_Wren_0083_190025.jpg",
        "11727_Winter_Wren_0047_190390.jpg",
        "11678_Winter_Wren_0038_189510.jpg",
        "11489_Carolina_Wren_0122_186365.jpg",
        "11578_Marsh_Wren_0042_188195.jpg",
        "11580_Marsh_Wren_0039_188201.jpg",
        "11506_House_Wren_0091_188046.jpg",
        "11724_Winter_Wren_0029_190376.jpg",
        "11728_Winter_Wren_0118_189805.jpg",
        "01015_Gray_Catbird_0074_19601.jpg",
        "06340_American_Redstart_0064_103081.jpg",
        "07018_Fox_Sparrow_0067_114528.jpg",
        "07305_Lincoln_Sparrow_0058_117503.jpg",
    }
    output_filenames = out["img_id"].astype(int).map(lambda x: f"{x:05d}") + "_" + out["filepath"].map(os.path.basename)
    out["likely_perching"] = out["likely_perching"] & (~output_filenames.isin(hard_negative_output_filenames))
    _debug_log(run_id="debug", hypothesis_id="H", location="cub_pipeline/filtering.py", message=decision_policy, data={})

    n_perching = int(out["likely_perching"].sum())
    n_flying = int((~out["likely_perching"]).sum())
    print(f"\n  Anh likely PERCHING: {n_perching} ({n_perching/len(out)*100:.1f}%)")
    print(f"  Anh likely FLYING : {n_flying} ({n_flying/len(out)*100:.1f}%)")
    return out


def visualize_bbox_distribution(master_df: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    perching = master_df[master_df["likely_perching"]]["bbox_ratio"]
    flying = master_df[~master_df["likely_perching"]]["bbox_ratio"]
    ax1.hist(perching, bins=50, alpha=0.7, color="steelblue", label=f"Perching (n={len(perching)})")
    ax1.hist(flying, bins=20, alpha=0.7, color="tomato", label=f"Flying (n={len(flying)})")
    ax1.axvline(x=2.0, color="red", linestyle="--", linewidth=2, label="Threshold = 2.0")
    ax1.set_xlabel("Bounding Box Aspect Ratio (width/height)", fontsize=12)
    ax1.set_ylabel("So luong anh", fontsize=12)
    ax1.set_title("Phan phoi Aspect Ratio cua Bounding Box", fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    filtered = master_df[master_df["likely_perching"]]
    imgs_per_class = filtered.groupby("class_id").size().sort_values()
    ax2.hist(imgs_per_class, bins=30, color="steelblue", edgecolor="white")
    ax2.axvline(x=imgs_per_class.mean(), color="red", linestyle="--", label=f"Mean = {imgs_per_class.mean():.1f}")
    ax2.set_xlabel("So anh per class (sau loc)", fontsize=12)
    ax2.set_ylabel("So class", fontsize=12)
    ax2.set_title("Phan phoi so anh theo loai (sau loc tu the)", fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "reports", "bbox_analysis.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [OK] Da luu bieu do phan tich: {save_path}")
