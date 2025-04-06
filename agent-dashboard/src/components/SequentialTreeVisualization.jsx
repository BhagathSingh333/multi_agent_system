// components/SequentialTreeVisualization.jsx
import { useEffect, useRef, useState } from 'react';

const SequentialTreeVisualization = ({ interactions, onThoughtProcessSelect, agentColors }) => {
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 0 });

  // Use provided agent colors or fallback to defaults
  const getAgentColor = (agentName) => {
    return agentColors?.[agentName] || {
      Planner: '#FF6B6B',
      Executor: '#4ECDC4',
      Researcher: '#45B7D1',
      Critic: '#FFA07A',
      Coordinator: '#F7B801',
      Verification: '#8B5CF6',
      Summary: '#EC4899',
    }[agentName] || '#6366F1';
  };

  useEffect(() => {
    // Update dimensions when container size changes
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width } = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: width - 40, // Account for padding
          height: 0 // Will be calculated based on node positions
        });
      }
    };

    window.addEventListener('resize', updateDimensions);
    updateDimensions();

    return () => window.removeEventListener('resize', updateDimensions);
  }, [interactions]);

  useEffect(() => {
    if (!interactions.length || !svgRef.current) return;

    // Clear previous content
    while (svgRef.current.firstChild) {
      svgRef.current.removeChild(svgRef.current.firstChild);
    }

    const { width } = dimensions;

    // Create a sequential tree structure
    const nodesByLevel = {}; // Dictionary where key is level, value is array of nodes at that level
    const nodesByNameAndLevel = {}; // Dictionary to track unique nodes by name at each level
    const nodes = []; // All nodes
    const edges = []; // All edges
    const nodeMap = {}; // Map of nodeId to node object

    // Start with the first interaction's source at level 0
    const firstSource = interactions[0].source;
    const firstNode = {
      id: `${firstSource}-0`,
      name: firstSource,
      level: 0,
      index: 0,
      thoughtProcess: interactions[0].thoughtProcess || "No thought process recorded for this agent."
    };

    nodesByLevel[0] = [firstNode];
    nodesByNameAndLevel[`0-${firstSource}`] = firstNode;
    nodes.push(firstNode);
    nodeMap[firstNode.id] = firstNode;

    // Process each interaction to build the tree
    interactions.forEach((interaction, interactionIndex) => {
      const { source, target, thoughtProcess } = interaction;

      // Find the latest occurrence of the source agent
      let sourceNode = null;
      for (let i = nodes.length - 1; i >= 0; i--) {
        if (nodes[i].name === source) {
          sourceNode = nodes[i];
          break;
        }
      }

      if (!sourceNode) {
        console.error(`Source node ${source} not found for interaction ${interactionIndex}`);
        return;
      }

      // Calculate target level
      const targetLevel = sourceNode.level + 1;

      // Check if a node with this name already exists at this level
      const existingNodeKey = `${targetLevel}-${target}`;
      let targetNode = nodesByNameAndLevel[existingNodeKey];

      if (!targetNode) {
        // Create a new target node if one doesn't exist at this level
        targetNode = {
          id: `${target}-${targetLevel}`,
          name: target,
          level: targetLevel,
          index: nodesByLevel[targetLevel] ? nodesByLevel[targetLevel].length : 0,
          thoughtProcess: interaction.thoughtProcess || "No thought process recorded for this agent."
        };

        // Add the target node to the appropriate level
        if (!nodesByLevel[targetLevel]) {
          nodesByLevel[targetLevel] = [];
        }
        nodesByLevel[targetLevel].push(targetNode);
        nodes.push(targetNode);
        nodeMap[targetNode.id] = targetNode;
        nodesByNameAndLevel[existingNodeKey] = targetNode;
      } else {
        // Merge thought processes if we're reusing a node
        if (interaction.thoughtProcess) {
          targetNode.thoughtProcess += "\n\n" + interaction.thoughtProcess;
        }
      }

      // Add edge between source and target
      edges.push({
        source: sourceNode.id,
        target: targetNode.id,
        sequenceNumber: interactionIndex + 1
      });
    });

    // Calculate node positions
    const levelHeight = 120;
    const levelWidth = width - 100;
    let maxY = 0;

    // Position nodes
    Object.entries(nodesByLevel).forEach(([level, levelNodes]) => {
      const nodeSpacing = levelWidth / (levelNodes.length + 1);

      levelNodes.forEach((node, index) => {
        node.x = 50 + ((index + 1) * nodeSpacing);
        node.y = 80 + (parseInt(level) * levelHeight);
        maxY = Math.max(maxY, node.y);
      });
    });

    // Set the SVG height based on the deepest node plus padding
    const height = maxY + 120; // Add padding for the last node

    // Set SVG dimensions
    svgRef.current.setAttribute('width', width);
    svgRef.current.setAttribute('height', height);
    svgRef.current.setAttribute('viewBox', `0 0 ${width} ${height}`);

    // Draw all connections
    edges.forEach(edge => {
      const sourceNode = nodeMap[edge.source];
      const targetNode = nodeMap[edge.target];

      if (!sourceNode || !targetNode) return;

      // Draw connection line
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');

      // Calculate control points for a curved path
      const midY = (sourceNode.y + targetNode.y) / 2;
      const pathData = `M ${sourceNode.x} ${sourceNode.y + 40} C ${sourceNode.x} ${midY}, ${targetNode.x} ${midY}, ${targetNode.x} ${targetNode.y - 40}`;

      line.setAttribute('d', pathData);
      line.setAttribute('stroke', getAgentColor(sourceNode.name));
      line.setAttribute('stroke-width', '2');
      line.setAttribute('fill', 'none');
      svgRef.current.appendChild(line);

      // Add sequence number
      const midX = (sourceNode.x + targetNode.x) / 2;

      const numberCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      numberCircle.setAttribute('cx', midX);
      numberCircle.setAttribute('cy', midY);
      numberCircle.setAttribute('r', '15');
      numberCircle.setAttribute('fill', 'white');
      numberCircle.setAttribute('stroke', getAgentColor(sourceNode.name));
      numberCircle.setAttribute('stroke-width', '1.5');
      svgRef.current.appendChild(numberCircle);

      const numberText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      numberText.setAttribute('x', midX);
      numberText.setAttribute('y', midY);
      numberText.setAttribute('text-anchor', 'middle');
      numberText.setAttribute('dominant-baseline', 'middle');
      numberText.setAttribute('font-size', '12');
      numberText.setAttribute('font-weight', 'bold');
      numberText.textContent = edge.sequenceNumber;
      svgRef.current.appendChild(numberText);
    });

    // Draw all nodes
    nodes.forEach(node => {
      // Create node group for click handling
      const nodeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      nodeGroup.setAttribute('cursor', 'pointer');
      nodeGroup.addEventListener('click', () => {
        if (onThoughtProcessSelect) {
          onThoughtProcessSelect(node);
        }
      });

      // Create node ellipse
      const ellipse = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
      ellipse.setAttribute('cx', node.x);
      ellipse.setAttribute('cy', node.y);
      ellipse.setAttribute('rx', 80); // Horizontal radius
      ellipse.setAttribute('ry', 40); // Vertical radius
      ellipse.setAttribute('fill', getAgentColor(node.name) || '#6366F1');
      nodeGroup.appendChild(ellipse);


      // Add agent name inside the circle
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', node.x);
      text.setAttribute('y', node.y);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('dominant-baseline', 'middle');
      text.setAttribute('fill', 'white');
      text.setAttribute('font-size', '12');
      text.setAttribute('font-weight', 'bold');
      text.textContent = node.name;
      nodeGroup.appendChild(text);

      // Add a hint that the node is clickable
      const clickHint = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      clickHint.setAttribute('cx', node.x + 30);
      clickHint.setAttribute('cy', node.y - 30);
      clickHint.setAttribute('r', 10);
      clickHint.setAttribute('fill', 'white');
      clickHint.setAttribute('stroke', getAgentColor(node.name));
      clickHint.setAttribute('stroke-width', '1.5');
      nodeGroup.appendChild(clickHint);

      const clickIcon = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      clickIcon.setAttribute('x', node.x + 30);
      clickIcon.setAttribute('y', node.y - 30);
      clickIcon.setAttribute('text-anchor', 'middle');
      clickIcon.setAttribute('dominant-baseline', 'middle');
      clickIcon.setAttribute('font-size', '10');
      clickIcon.setAttribute('font-weight', 'bold');
      clickIcon.textContent = 'i';
      nodeGroup.appendChild(clickIcon);

      svgRef.current.appendChild(nodeGroup);
    });

  }, [interactions, dimensions, onThoughtProcessSelect, agentColors]);

  return (
    <div ref={containerRef} className="w-full bg-white rounded-xl p-4">
      <svg ref={svgRef} className="w-full" style={{ height: 'auto' }}></svg>
    </div>
  );
};

export default SequentialTreeVisualization;
