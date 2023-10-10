# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    GreinerHormann()

The Greiner-Hormann algorithm for clipping polygons.

## References

* Greiner, G. & Hormann, K. 1998. [Efficient Clipping of Arbitrary Polygons]
  (https://dl.acm.org/doi/pdf/10.1145/274363.274364)
"""

struct GreinerHormann <: ClippingMethod end

import Makie, GLMakie
GLMakie.activate!()

mutable struct ghVertex
  pos::Point{2, Float32} # Point of vertex
  next::Point{2, Float32} # Point of next vertex
  prev::Point{2, Float32} # Point of previous vertex
  # Point of first vertex of next resulting
  # clipped polygon, in case there are multiple
  # nextPoly::Point
  intersect::Bool # Whether point is an intersection
  # If intersection, whether it enters or exit other polygon
  entryExit::Symbol
  # neighbor::Point # Switch polygon being analyzed
  alphaRing::Float32 # Normalized position along edge, if intersection point
  alphaOther::Float32 # Normalized position along edge, if intersection point
end

# Get value in correct alpha field according to polygon
getAlpha(p::ghVertex, polName::Symbol)::Float32 = polName == :ring ? p.alphaRing : p.alphaOther

function plotPol(pol)
  fig = Makie.Figure()
  xCoords = [v.coords[1] for v in vertices(pol)]
  yCoords = [v.coords[2] for v in vertices(pol)]
  # ax = Makie.Axis(fig[1, 1]; limits = (-2.5, 8, -2.5, 8))
  ax = Makie.Axis(fig[1, 1]; limits = (
    minimum(xCoords) - 0.5, maximum(xCoords) + 0.5,
    minimum(yCoords) - 0.5, maximum(yCoords) + 0.5
  ))
  coords = NTuple{2, Float32}[]
  for v in pol.vertices
    push!(coords, Tuple([v.coords[i] for i in 1:2]))
  end
  Makie.scatter!(ax, coords)
  Makie.text!(
    coords, text = ["$i  " for i in 1:length(pol.vertices)],
    align = (:right, :baseline),
    fontsize = 23
  )
  push!(coords, Tuple(pol.vertices[1].coords))
  Makie.poly!(ax, coords; color = :transparent, strokecolor = :black, strokewidth = 2)
  Makie.display(fig)
end

# Include new intersection in polygon vertex list
function addPoint!(
  polData::Dict{Symbol, Dict{Symbol, Vector}}, pol::Symbol, newID::Int,
  newPoint, alphas::NTuple{2, Real}, startEdge::Bool
)
  if newID == length(polData[pol][:list])
    append!( # Intersection is new last vertex of polygon
      polData[pol][:list],
      [
        ghVertex(
          newPoint,
          polData[pol][:list][1].pos,
          polData[pol][:list][end].pos,
          true, :entry, alphas...
        )
      ]
    )
    newID += 1
  else
    startEdge && (newID += 1)
    insert!(
      polData[pol][:list],
      newID,
      ghVertex(
        newPoint,
        polData[pol][:list][newID].pos,
        polData[pol][:list][newID - 1].pos,
        true, :entry, alphas...
      )
    )
  end
  # update 'next' and 'prev' fields of vertices that
  # are adjacent on the list to the one just inserted/appended
  polData[pol][:list][ # 'next' field of previous point
    newID == 1 ? length(polData[pol][:list]) : newID - 1
  ].next = polData[pol][:list][newID].pos
  polData[pol][:list][ # 'prev' field of next point
    newID == length(polData[pol][:list]) ? 1 : newID + 1
  ].prev = polData[pol][:list][newID].pos
  nothing
end

# Check which vertices of one polygon are inside the other. Then
# use that information to classify intersections between 'entry' or 'exit'
function entryExit!(_ring::Ring, _other::Ring, polData::Dict{Symbol, Dict{Symbol, Vector}})
  for (pol, augList) in zip([_other, _ring], [polData[:ring][:list], polData[:other][:list]])
    for polVertexID in 2 : length(augList) - 1
      if augList[polVertexID].intersect == true # intersection point
        # if only the start of the edge is in the other polygon,
        # this intersection is an exit
        if augList[polVertexID - 1].pos ∈ pol && !(augList[polVertexID + 1].pos ∈ pol)
          augList[polVertexID].entryExit = :exit
        else
          augList[polVertexID].entryExit = :entry
        end
      end
    end
  end
end

# Insert new intersection in the case of there already being
# previous intersections in the same edge
function insertWithPrevIntersec!(
  otherInters::Vector{NTuple{2, Union{Int64, Meshes.ghVertex}}}, pol::Symbol, intersec::Point,
  ringAlpha::Float32, otherAlpha::Float32, polData::Dict{Symbol, Dict{Symbol, Vector}}
)
  alpha = pol == :ring ? ringAlpha : otherAlpha
  # Sort intersections in current edge by alpha values
  sort!(otherInters, by = h -> getAlpha(h[2], pol))
  prevInterID = otherInters[end][1] + 1
  for (pointID, otherInterPoint) in otherInters
    # Look for last intersection before the one just found. It's ID
    # indicates where in the list of vertices to insert new intersection
    if getAlpha(otherInterPoint, pol) > alpha
      prevInterID = pointID - 1
      break
    end
  end
  addPoint!(polData, pol, prevInterID, intersec, (ringAlpha, otherAlpha), false)
end

function clip(ring::Ring{Dim,T}, other::Ring{Dim,T}, ::GreinerHormann) where {Dim,T}
  Makie.display(Makie.Figure())
  plotPol(ring)
  plotPol(other)
  function findPoint(p::Point2f, polData::Dict)::Union{Int, Nothing}
    return findfirst(≈(p), [el.pos for el in polData[:list]])
  end
  # Polygon edges
  ringEdges = [
    Segment(
      ring.vertices[i],
      ring.vertices[i % length(ring.vertices) + 1]
    ) for i in 1:length(ring.vertices)
  ]
  otherEdges = [
    Segment(
      other.vertices[i],
      other.vertices[i % length(other.vertices) + 1]
    ) for i in 1:length(other.vertices)
  ]
  # Check for degenerate case of a vertex being on top of an
  # edge of the other polygon
  for (poly, vertexEnumList, edgeList) in zip(
    [:other, :ring],
    enumerate.([other.vertices, ring.vertices]),
    [ringEdges, otherEdges]
  )
    for ((vertID, vert), polEdge) in Iterators.product(vertexEnumList, edgeList)
      
      if vert ∈ Line(polEdge.vertices...)
        # Slightly perturb position
        perturbScale = 1e3
        if poly == :ring
          ring.vertices[vertID] = Point(
            ring.vertices[vertID].coords + Vec2f(rand() / perturbScale, rand() / perturbScale)
          )
        else
          other.vertices[vertID] = Point(
            other.vertices[vertID].coords + Vec2f(rand() / perturbScale, rand() / perturbScale)
          )
        end
      end
    end
  end
  polygonData = Dict( # data needed
    :ring => Dict(
      :list => [
        ghVertex(
          polVertex,
          ring.vertices[vertexID % length(ring.vertices) + 1],
          repeat(ring.vertices, 2)[
            collect(length(ring.vertices) : length(ring.vertices) * 2)[vertexID]
          ], false, :entry, 0f0, 0f0
        ) for (vertexID, polVertex) in enumerate(ring.vertices)
      ],
      :edges => [
        Segment(
          ring.vertices[i],
          ring.vertices[i % length(ring.vertices) + 1]
        ) for i in 1:length(ring.vertices)
      ]
    ),
    :other => Dict(
      :list => [
        ghVertex(
          polVertex,
          other.vertices[vertexID % length(other.vertices) + 1],
          repeat(other.vertices, 2)[
            collect(length(other.vertices) : length(other.vertices) * 2)[vertexID]
          ],
          false, :entry, 0f0, 0f0
        ) for (vertexID, polVertex) in enumerate(other.vertices)
      ],
      :edges => [
        Segment(
          other.vertices[i],
          other.vertices[i % length(other.vertices) + 1]
        ) for i in 1:length(other.vertices)
      ]
    )
  )
  ## Phase 1: add intersections between edges to vertex lists
  intersections = 0 # Count intersections
  # Iterate in pairs of edges of both polygons
  for (ringEdge, otherEdge) in Iterators.product(polygonData[:ring][:edges], polygonData[:other][:edges])
    # Get intersection between current pair of edges
    interPoint = intersection(ringEdge, otherEdge) |> get
    if interPoint !== nothing # If there's an intersection
      intersections += 1 # Count intersection
      # Determine normalized position of intersection in both edges
      alphaRing = length(Segment(minimum(ringEdge), interPoint)) / length(ringEdge)
      alphaOther = length(Segment(minimum(otherEdge), interPoint)) / length(otherEdge)
      # Look for previous intersections in the same edges
      for (pol, polEdge, alpha) in zip([:ring, :other], [ringEdge, otherEdge], [alphaRing, alphaOther])
        previousInter = NTuple{2, Union{Int, ghVertex}}[]
        for polPoint in enumerate(polygonData[pol][:list])
          # Change type for tolerance in edge verification
          line = Line(polEdge.vertices...)
          (polPoint[2].pos ∈ line && polPoint[2].intersect == true) && push!(previousInter, polPoint)
        end
        # If there are previous intersections in current edge
        if any([getAlpha(h[2], pol) for h in previousInter] .< alpha)
          insertWithPrevIntersec!(
            previousInter, pol, interPoint, alphaRing, alphaOther, polygonData
          )
        else
          edgeStart = findPoint(==(minimum(polEdge)).x, polygonData[pol])
          # Include intersection in vertices lists of polygons
          addPoint!(polygonData, pol, edgeStart, interPoint, (alphaRing, alphaOther), true)
        end
      end
    end
  end
  plotPol(Ring([v.pos for v in polygonData[:ring][:list]]))
  plotPol(Ring([v.pos for v in polygonData[:other][:list]]))
  # Phase 2: classify intersections in either "entry" or "exit"
  entryExit!(ring, other, polygonData)
  # Phase 3: visit vertices and get clipped polygon
  intersectionsVisited = 1
  # Start at intersection in 'ring'
  currentPoint = polygonData[:ring][:list][
    findfirst(b -> b.intersect == true && b.entryExit == :entry, polygonData[:ring][:list])
  ]
  currentPolygon = :ring
  clippedPolygon = [currentPoint]
  # Start at an intersection point. Follow edge of 'ring' towards the inside
  # of 'other'. Save sequence of vertices visited. At next intersection, change
  # direction to follow edge of 'other'. Follow this procedure until starting
  # intersection is revisited. If the result of clipping is a set of
  # polygons, some intersection points will remain unvisited. Repeat process
  # starting at a unvisited intersection point untill they are all visited.
  iters = 0
  # while iters < 20
  while intersectionsVisited <= intersections
    iters += 1
    if currentPoint.intersect == true && currentPoint.entryExit == :exit
      # Update current polygon
      currentPolygon = currentPolygon == :ring ? :other : :ring
      # Go to next point in new polygon
      currentPoint = polygonData[currentPolygon][:list][
        findPoint(currentPoint.pos, polygonData[currentPolygon]) + 1
      ]
      # Include new point in clipped polygon
      if findPoint(currentPoint.pos, Dict(:list => clippedPolygon)) === nothing
        push!(clippedPolygon, currentPoint)
      end
      intersectionsVisited += 1
    else
      # Go to next point in polygon
      currentPoint = polygonData[currentPolygon][:list][
        findPoint(currentPoint.next, polygonData[currentPolygon])
      ]
      # Include new point in clipped polygon
      if findPoint(currentPoint.pos, Dict(:list => clippedPolygon)) === nothing
        push!(clippedPolygon, currentPoint)
      end
      if currentPoint.intersect == true
        intersectionsVisited += 1 
      end
    end
  end
  plotPol(Ring([v.pos for v in clippedPolygon]))
  isempty(clippedPolygon) ? nothing : Ring([p.pos for p in clippedPolygon])
end