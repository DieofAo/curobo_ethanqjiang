#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a prefixed Marvin + JAKA composite URDF for RViz."""

import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET


def parse_urdf(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    end_tag = "</robot>"
    end = text.find(end_tag)
    if end >= 0:
        text = text[: end + len(end_tag)]
    return ET.fromstring(text)


def make_abs_mesh_path(filename, urdf_dir):
    if filename.startswith("file://") or "://" in filename:
        return filename
    abs_path = os.path.abspath(os.path.join(urdf_dir, filename))
    return "file://" + abs_path


def prefix_robot(source_root, prefix, urdf_path):
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    prefixed = []

    for elem in list(source_root):
        if elem.tag not in ("link", "joint", "material", "transmission", "gazebo"):
            continue
        elem = copy.deepcopy(elem)

        if "name" in elem.attrib and elem.attrib["name"]:
            elem.attrib["name"] = prefix + elem.attrib["name"]

        for ref in elem.iter():
            if ref.tag in ("parent", "child") and "link" in ref.attrib:
                ref.attrib["link"] = prefix + ref.attrib["link"]
            elif ref.tag == "mimic" and "joint" in ref.attrib:
                ref.attrib["joint"] = prefix + ref.attrib["joint"]
            elif ref.tag == "mesh" and "filename" in ref.attrib:
                ref.attrib["filename"] = make_abs_mesh_path(ref.attrib["filename"], urdf_dir)
            elif ref.tag == "material" and "name" in ref.attrib and ref.attrib["name"]:
                ref.attrib["name"] = prefix + ref.attrib["name"]

        prefixed.append(elem)

    return prefixed


def add_fixed_joint(root, name, parent, child, xyz, rpy):
    joint = ET.SubElement(root, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})


def main():
    parser = argparse.ArgumentParser(description="Generate Marvin + JAKA composite URDF")
    parser.add_argument("--marvin-urdf", required=True)
    parser.add_argument("--jaka-urdf", required=True)
    parser.add_argument("--marvin-xyz", default="-0.6 0 0")
    parser.add_argument("--jaka-xyz", default="0.6 0 0")
    parser.add_argument("--marvin-rpy", default="0 0 0")
    parser.add_argument("--jaka-rpy", default="0 0 0")
    args = parser.parse_args()

    marvin = parse_urdf(args.marvin_urdf)
    jaka = parse_urdf(args.jaka_urdf)

    root = ET.Element("robot", {"name": "marvin_jaka_dual"})
    ET.SubElement(root, "link", {"name": "world"})

    for elem in prefix_robot(marvin, "marvin_", args.marvin_urdf):
        root.append(elem)
    for elem in prefix_robot(jaka, "jaka_", args.jaka_urdf):
        root.append(elem)

    add_fixed_joint(
        root,
        "world_to_marvin_robot_stand",
        "world",
        "marvin_robot_stand",
        args.marvin_xyz,
        args.marvin_rpy,
    )
    add_fixed_joint(
        root,
        "world_to_jaka_LINK_BASE",
        "world",
        "jaka_LINK_BASE",
        args.jaka_xyz,
        args.jaka_rpy,
    )

    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")
    sys.stdout.write(ET.tostring(root, encoding="unicode"))


if __name__ == "__main__":
    main()
