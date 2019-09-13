for f in *.ply; do
  pcl_converter -f ascii ./"$f" ./"${f%.ply}.pcd"
done
mv *.pcd ../pcd/
