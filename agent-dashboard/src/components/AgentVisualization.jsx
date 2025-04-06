// components/AgentVisualization.jsx
import { useRef, useEffect, useState } from 'react'
import { useSpring, animated } from '@react-spring/web'

const AgentVisualization = ({ interactions, agentColors, onEdgeClick, highlightedStep }) => {
  const [agents, setAgents] = useState({})
  const [connectionPaths, setConnectionPaths] = useState({})
  const [hoveredPath, setHoveredPath] = useState(null)
  const [hoveredNode, setHoveredNode] = useState(null)
  const containerRef = useRef(null)

  // Get color for an agent, using the passed-in agentColors prop or fallback to random
  const getAgentColor = (agentName) => {
    return agentColors && agentColors[agentName]
      ? agentColors[agentName]
      : getRandomColor(agentName);
  }

  // Get a random color for agents without predefined colors
  const getRandomColor = (agentName) => {
    const colors = {
      "Crop Recommender Agent": '#FF6B6B',     // Using Planner color
      "Environmental Information Agent": '#45B7D1',   // Using Researcher color
      "Marketing Information Agent": '#4ECDC4',       // Using Executor color
      "Soil Information Agent": '#FFA07A',      // Using Critic color
      "Verification Agent": '#F7B801',        // Using Coordinator color
      "Summary Agent": '#8B5CF6',             // New color for Summary Agent

      // Frontend friendly names
      "Crop Recommender": '#FF6B6B',
      "Environmental Info": '#45B7D1',
      "Marketing Info": '#4ECDC4',
      "Soil Health": '#FFA07A',
      "Verification": '#F7B801',
      "Summary": '#8B5CF6',

      // User for the final interaction
      "User": '#6366F1',

      // Backend agent abbreviated names as fallback
      CropRecommenderAgent: '#FF6B6B',
      EnvironmentalInfoAgent: '#45B7D1',
      MarketingInfoAgent: '#4ECDC4',
      SoilHealthInfoAgent: '#FFA07A',
      VerificationAgent: '#F7B801',
      SummaryAgent: '#8B5CF6',
    }
    return colors[agentName]
  }

  // Initialize agents with positions
  useEffect(() => {
    if (!containerRef.current) return

    const width = containerRef.current.clientWidth
    const height = containerRef.current.clientHeight

    // Extract all unique agent names from interactions
    const agentNames = new Set()
    interactions.forEach(interaction => {
      agentNames.add(interaction.source)
      agentNames.add(interaction.target)
    })

    // Create agent objects with positions
    const agentObj = {}
    const centerX = width / 2
    const centerY = height / 2
    const radius = Math.min(width, height) * 0.35

    Array.from(agentNames).forEach((name, index) => {
      const angle = (index / agentNames.size) * 2 * Math.PI
      agentObj[name] = {
        id: name,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
        color: getAgentColor(name),
      }
    })

    setAgents(agentObj)

    // Generate connection paths
    const paths = {}

    // Count interactions between each pair of agents
    const interactionCounts = {}
    const pairInteractionIndices = {}

    interactions.forEach(interaction => {
      const key = `${interaction.source}-${interaction.target}`
      interactionCounts[key] = (interactionCounts[key] || 0) + 1

      // Track indices of interactions between the same pair
      if (!pairInteractionIndices[key]) {
        pairInteractionIndices[key] = []
      }
      pairInteractionIndices[key].push(interactions.indexOf(interaction))
    })

    // Generate curved paths for each interaction
    interactions.forEach((interaction, index) => {
      const { source, target } = interaction
      const key = `${source}-${target}`
      const reverseKey = `${target}-${source}`

      // Only create paths for agent pairs that exist in our agents object
      if (agentObj[source] && agentObj[target]) {
        const sourceX = agentObj[source].x
        const sourceY = agentObj[source].y
        const targetX = agentObj[target].x
        const targetY = agentObj[target].y

        // Calculate midpoint
        const midX = (sourceX + targetX) / 2
        const midY = (sourceY + targetY) / 2

        // Calculate normal vector for curve control point
        const dx = targetX - sourceX
        const dy = targetY - sourceY
        const length = Math.sqrt(dx * dx + dy * dy)
        const normalX = -dy / length
        const normalY = dx / length

        // Determine curve direction based on whether it's a response
        // If there are interactions in both directions, curve them differently
        const hasBidirectional = interactionCounts[key] > 0 && interactionCounts[reverseKey] > 0

        // Find which number interaction this is between this pair
        const pairIndex = pairInteractionIndices[key].indexOf(index)

        // Calculate curve intensity based on interaction count and pair index
        const baseIntensity = hasBidirectional ? 60 : 40
        const curveIntensity = baseIntensity + Math.min(pairIndex, 5) * 30

        // Alternate curve direction for consecutive interactions with more variation
        const alternateDirection = pairIndex % 2 === 0 ? 1 : -1.2

        // Calculate control point
        const controlX = midX + normalX * curveIntensity * alternateDirection
        const controlY = midY + normalY * curveIntensity * alternateDirection

        // Create SVG path
        const path = `M ${sourceX} ${sourceY} Q ${controlX} ${controlY}, ${targetX} ${targetY}`

        // Calculate position for sequence number (along the path)
        const sequenceX = midX + normalX * (curveIntensity * 0.8) * alternateDirection
        const sequenceY = midY + normalY * (curveIntensity * 0.8) * alternateDirection

        paths[`${index}`] = {
          path,
          source,
          target,
          controlPoint: { x: controlX, y: controlY },
          sequencePosition: { x: sequenceX, y: sequenceY }
        }
      }
    })

    setConnectionPaths(paths)
  }, [interactions, agentColors])

  // Calculate point along a quadratic bezier curve
  const getPointOnQuadraticCurve = (startX, startY, controlX, controlY, endX, endY, t) => {
    const x = Math.pow(1 - t, 2) * startX + 2 * (1 - t) * t * controlX + Math.pow(t, 2) * endX
    const y = Math.pow(1 - t, 2) * startY + 2 * (1 - t) * t * controlY + Math.pow(t, 2) * endY
    return { x, y }
  }

  // Calculate angle at a point on the curve (for arrow direction)
  const getAngleOnQuadraticCurve = (startX, startY, controlX, controlY, endX, endY, t) => {
    // Derivative of quadratic bezier
    const dx = 2 * (1 - t) * (controlX - startX) + 2 * t * (endX - controlX)
    const dy = 2 * (1 - t) * (controlY - startY) + 2 * t * (endY - controlY)
    return Math.atan2(dy, dx)
  }

  // Always call useSpring at the top level, regardless of conditions
  const springProps = useSpring({
    from: { progress: 0, opacity: 0 },
    to: {
      progress: interactions.length > 0 ? 1 : 0,
      opacity: interactions.length > 0 ? 1 : 0
    },
    config: { tension: 120, friction: 14 },
    delay: 100
  })

  return (
    <div
      ref={containerRef}
      className="w-full bg-slate-50 rounded-xl overflow-hidden border border-slate-200 relative"
      style={{
        height: `${Math.max(400, 300 + interactions.length * 10)}px`
      }}
    >
      <svg className="absolute inset-0 w-full h-full">
        {/* Draw all communication paths */}
        {Object.entries(connectionPaths).map(([index, pathData]) => {
          const sourceAgent = agents[pathData.source]
          const targetAgent = agents[pathData.target]

          if (!sourceAgent || !targetAgent) return null

          const interactionIndex = parseInt(index)
          const isActive = interactionIndex === interactions.length - 1
          const isHighlighted = highlightedStep === interactionIndex + 1
          const isHovered = hoveredPath === index

          // Calculate the point near the target for the arrowhead
          const t = 0.95 // Position along the curve (0-1)
          const arrowPoint = getPointOnQuadraticCurve(
            sourceAgent.x, sourceAgent.y,
            pathData.controlPoint.x, pathData.controlPoint.y,
            targetAgent.x, targetAgent.y,
            t
          )

          // Calculate midpoint for the mid-path arrow (at 50% of the path)
          const midPoint = getPointOnQuadraticCurve(
            sourceAgent.x, sourceAgent.y,
            pathData.controlPoint.x, pathData.controlPoint.y,
            targetAgent.x, targetAgent.y,
            0.5
          )

          // Calculate angle for the arrowheads
          const angle = getAngleOnQuadraticCurve(
            sourceAgent.x, sourceAgent.y,
            pathData.controlPoint.x, pathData.controlPoint.y,
            targetAgent.x, targetAgent.y,
            t
          )

          const midAngle = getAngleOnQuadraticCurve(
            sourceAgent.x, sourceAgent.y,
            pathData.controlPoint.x, pathData.controlPoint.y,
            targetAgent.x, targetAgent.y,
            0.5
          )

          return (
            <g
              key={`path-${index}`}
              onMouseEnter={() => setHoveredPath(index)}
              onMouseLeave={() => setHoveredPath(null)}
              onClick={() => onEdgeClick && onEdgeClick(interactions[interactionIndex])}
              style={{ cursor: 'pointer' }}
            >
              {/* Path line */}
              <path
                d={pathData.path}
                fill="none"
                stroke={isHovered || isHighlighted ? sourceAgent.color : isActive ? sourceAgent.color : "#CBD5E1"}
                strokeWidth={isHovered || isHighlighted ? 3 : isActive ? 2.5 : 1.5}
                strokeOpacity={isHovered || isHighlighted ? 1 : isActive ? 1 : 0.7}
                strokeDasharray={isHovered || isHighlighted ? "none" : isActive ? "none" : "none"}
              />

              {/* Arrowhead at the end */}
              <polygon
                points="0,-6 12,0 0,6"
                fill={isHovered || isHighlighted ? sourceAgent.color : isActive ? sourceAgent.color : "#CBD5E1"}
                opacity={isHovered || isHighlighted ? 1 : isActive ? 1 : 0.7}
                transform={`translate(${arrowPoint.x},${arrowPoint.y}) rotate(${angle * 180 / Math.PI})`}
              />

              {/* Mid-path arrow showing direction */}
              <polygon
                points="0,-5 10,0 0,5"
                fill={isHovered || isHighlighted ? sourceAgent.color : isActive ? sourceAgent.color : "#CBD5E1"}
                opacity={isHovered || isHighlighted ? 1 : isActive ? 1 : 0.7}
                transform={`translate(${midPoint.x},${midPoint.y}) rotate(${midAngle * 180 / Math.PI})`}
              />

              {/* Sequence number */}
              <g transform={`translate(${pathData.sequencePosition.x},${pathData.sequencePosition.y})`}>
                <circle
                  r="10"
                  fill={isHovered || isHighlighted ? sourceAgent.color : isActive ? sourceAgent.color : "white"}
                  stroke={isHovered || isHighlighted ? "white" : isActive ? "white" : sourceAgent.color}
                  strokeWidth="1.5"
                />
                <text
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize="10"
                  fontWeight="bold"
                  fill={isHovered || isHighlighted ? "white" : isActive ? "white" : sourceAgent.color}
                >
                  {parseInt(index) + 1}
                </text>
              </g>

              {/* Tooltip that appears on hover */}
              {isHovered && (
                <g transform={`translate(${midPoint.x},${midPoint.y - 30})`}>
                  <rect
                    x="-100"
                    y="-15"
                    width="200"
                    height="30"
                    rx="5"
                    fill="white"
                    stroke={sourceAgent.color}
                    strokeWidth="1"
                    filter="drop-shadow(0px 1px 2px rgba(0,0,0,0.1))"
                  />
                  <text
                    x="0"
                    y="0"
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize="11"
                    fill="#4B5563"
                  >
                    {`${pathData.source} → ${pathData.target} (${parseInt(index) + 1})`}
                  </text>
                </g>
              )}
            </g>
          )
        })}

        {/* Animated message for the latest interaction */}
        {interactions.length > 0 && (() => {
          const latestInteraction = interactions[interactions.length - 1]
          const pathData = connectionPaths[`${interactions.length - 1}`]

          if (!pathData) return null

          const sourceAgent = agents[latestInteraction.source]
          const targetAgent = agents[latestInteraction.target]

          if (!sourceAgent || !targetAgent) return null

          return (
            <g>
              {/* Moving message dot */}
              <animated.circle
                cx={springProps.progress.to(p => {
                  const point = getPointOnQuadraticCurve(
                    sourceAgent.x, sourceAgent.y,
                    pathData.controlPoint.x, pathData.controlPoint.y,
                    targetAgent.x, targetAgent.y,
                    p
                  )
                  return point.x
                })}
                cy={springProps.progress.to(p => {
                  const point = getPointOnQuadraticCurve(
                    sourceAgent.x, sourceAgent.y,
                    pathData.controlPoint.x, pathData.controlPoint.y,
                    targetAgent.x, targetAgent.y,
                    p
                  )
                  return point.y
                })}
                r={5}
                fill={sourceAgent.color}
                opacity={springProps.opacity}
              />

              {/* Message tooltip */}
              <animated.g
                opacity={springProps.opacity}
                transform={springProps.progress.to(p => {
                  const point = getPointOnQuadraticCurve(
                    sourceAgent.x, sourceAgent.y,
                    pathData.controlPoint.x, pathData.controlPoint.y,
                    targetAgent.x, targetAgent.y,
                    p
                  )
                  return `translate(${point.x},${point.y - 20})`
                })}
              >
                <rect
                  x="-60"
                  y="-15"
                  width="120"
                  height="20"
                  rx="5"
                  fill="white"
                  stroke={sourceAgent.color}
                  strokeWidth="1"
                />
                <text
                  x="0"
                  y="0"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize="10"
                  fill="#4B5563"
                >
                  {latestInteraction.message?.substring(0, 20)}
                  {latestInteraction.message?.length > 20 ? '...' : ''}
                </text>
              </animated.g>
            </g>
          )
        })()}
      </svg>

      {/* Agent nodes */}
      {Object.values(agents).map((agent) => (
        <div
          key={agent.id}
          className="absolute transform -translate-x-1/2 -translate-y-1/2 flex flex-col items-center"
          style={{
            left: agent.x,
            top: agent.y,
            zIndex: 10 // Ensure agents are above paths
          }}
          onMouseEnter={() => setHoveredNode(agent.id)}
          onMouseLeave={() => setHoveredNode(null)}
        >
          <div
            className="w-24 h-24 rounded-full flex items-center justify-center text-white font-bold shadow-lg"
            style={{ backgroundColor: agent.color }}
          >
            <span className="text-xs text-center">{agent.id}</span>
          </div>
          {/* Node tooltip - only show on hover */}
          {hoveredNode === agent.id && (
            <div
              className="absolute -top-16 left-1/2 transform -translate-x-1/2 bg-white p-2 rounded-lg shadow-md border border-slate-200 z-20 w-48"
            >
              <p className="text-xs text-slate-700">
                For in-depth analysis of {agent.id} at a particular step, click the "View In-depth Agent Analysis" button below.
              </p>
            </div>
          )}
        </div>
      ))}

      {/* Legend for sequence numbers */}
      <div className="absolute bottom-2 right-2 bg-white/90 p-2 rounded-lg shadow-sm border border-slate-200 text-xs">
        <div className="font-semibold text-slate-700 mb-1">Communication Sequence</div>
        <div className="flex items-center">
          <div className="w-5 h-5 rounded-full bg-indigo-500 flex items-center justify-center text-white text-xs mr-2">1</div>
          <span>First interaction</span>
        </div>
        <div className="flex items-center mt-1">
          <div className="w-5 h-5 rounded-full bg-white border border-indigo-500 flex items-center justify-center text-indigo-500 text-xs mr-2">{interactions.length}</div>
          <span>Latest interaction</span>
        </div>
      </div>
    </div>
  )
}

export default AgentVisualization
