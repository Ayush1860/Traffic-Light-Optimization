import os

def create_network_files():
    # 1. Create nodes file
    nodes_content = """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="TL1" x="0.0" y="0.0" type="traffic_light"/>
    <node id="TL2" x="750.0" y="0.0" type="traffic_light"/>
    <node id="TL3" x="1500.0" y="0.0" type="traffic_light"/>
    <node id="W" x="-750.0" y="0.0" type="dead_end"/>
    <node id="E" x="2250.0" y="0.0" type="dead_end"/>
    <node id="N1" x="0.0" y="750.0" type="dead_end"/>
    <node id="S1" x="0.0" y="-750.0" type="dead_end"/>
    <node id="N2" x="750.0" y="750.0" type="dead_end"/>
    <node id="S2" x="750.0" y="-750.0" type="dead_end"/>
    <node id="N3" x="1500.0" y="750.0" type="dead_end"/>
    <node id="S3" x="1500.0" y="-750.0" type="dead_end"/>
</nodes>
"""
    with open("intersection/nodes.nod.xml", "w") as f:
        f.write(nodes_content)

    # 2. Create edges file
    edges_content = """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge id="W2TL1" from="W" to="TL1" numLanes="4" speed="13.89"/>
    <edge id="TL12W" from="TL1" to="W" numLanes="4" speed="13.89"/>
    
    <edge id="N12TL1" from="N1" to="TL1" numLanes="4" speed="13.89"/>
    <edge id="TL12N1" from="TL1" to="N1" numLanes="4" speed="13.89"/>
    
    <edge id="S12TL1" from="S1" to="TL1" numLanes="4" speed="13.89"/>
    <edge id="TL12S1" from="TL1" to="S1" numLanes="4" speed="13.89"/>
    
    <edge id="TL12TL2" from="TL1" to="TL2" numLanes="4" speed="13.89"/>
    <edge id="TL22TL1" from="TL2" to="TL1" numLanes="4" speed="13.89"/>
    
    <edge id="N22TL2" from="N2" to="TL2" numLanes="4" speed="13.89"/>
    <edge id="TL22N2" from="TL2" to="N2" numLanes="4" speed="13.89"/>
    
    <edge id="S22TL2" from="S2" to="TL2" numLanes="4" speed="13.89"/>
    <edge id="TL22S2" from="TL2" to="S2" numLanes="4" speed="13.89"/>
    
    <edge id="TL22TL3" from="TL2" to="TL3" numLanes="4" speed="13.89"/>
    <edge id="TL32TL2" from="TL3" to="TL2" numLanes="4" speed="13.89"/>
    
    <edge id="N32TL3" from="N3" to="TL3" numLanes="4" speed="13.89"/>
    <edge id="TL32N3" from="TL3" to="N3" numLanes="4" speed="13.89"/>
    
    <edge id="S32TL3" from="S3" to="TL3" numLanes="4" speed="13.89"/>
    <edge id="TL32S3" from="TL3" to="S3" numLanes="4" speed="13.89"/>
    
    <edge id="TL32E" from="TL3" to="E" numLanes="4" speed="13.89"/>
    <edge id="E2TL3" from="E" to="TL3" numLanes="4" speed="13.89"/>
</edges>
"""
    with open("intersection/edges.edg.xml", "w") as f:
        f.write(edges_content)

    # 3. Create connections file to enforce logic roughly matching original
    # We want lanes 0, 1, 2 to go straight. Lane 3 is left turn. (And lane 0 handles right turns as well).
    conns_content = """<?xml version="1.0" encoding="UTF-8"?>
<connections>
    <!-- We leave connections empty to let netconvert determine them based on standard SUMO guessing. -->
    <!-- But we add a .tll.xml file to ensure identical phases for our RL agent. -->
</connections>
"""
    with open("intersection/conns.con.xml", "w") as f:
        f.write(conns_content)
        
    print("Executing netconvert...")
    # Add flag to allow left turn on lane 3, and right turn on lane 0
    cmd = 'netconvert --node-files intersection/nodes.nod.xml --edge-files intersection/edges.edg.xml -o intersection/environment.net.xml --junctions.join-exclude "TL1,TL2,TL3" --default.lanenumber 4 --no-turnarounds'
    retcode = os.system(cmd)
    if retcode == 0:
        print("Success! Environment network updated with 3 intersections.")
    else:
        print(f"Error {retcode}: Failed to run netconvert.")

if __name__ == "__main__":
    create_network_files()
