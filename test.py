import uproot
import awkward as ak

file = uproot.open("data/TT_PU200/l1Nano_merged.root")

# check the tree name if unsure
print(file.keys())

tree = file["Events"]   # most L1Nano files use "Events"

gen_pt = tree["GenJet_pt"].array(library="ak")

# flatten all jets across all events
gen_pt_flat = ak.flatten(gen_pt)

print("Minimum GenJet pT:", ak.min(gen_pt_flat))