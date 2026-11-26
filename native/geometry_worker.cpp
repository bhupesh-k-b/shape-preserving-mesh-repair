// Isolated CGAL checks and surface-preserving triangle-soup refinement.
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/autorefinement.h>
#include <CGAL/Polygon_mesh_processing/distance.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Polygon_set_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/convex_hull_2.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>

using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Point = Kernel::Point_3;
using Triangle = std::array<std::size_t, 3>;
namespace PMP = CGAL::Polygon_mesh_processing;
struct Soup { std::vector<Point> points; std::vector<Triangle> triangles; };

Soup read(const std::string& path) {
  std::ifstream f(path);
  std::string magic; std::size_t nv, nf, ne;
  if (!(f >> magic >> nv >> nf >> ne) || magic != "OFF" || !nv || !nf)
    throw std::runtime_error("Invalid or empty OFF input");
  Soup soup; soup.points.reserve(nv); soup.triangles.reserve(nf);
  for (std::size_t i=0; i<nv; ++i) {
    double x,y,z; if (!(f >> x >> y >> z)) throw std::runtime_error("Truncated vertices");
    soup.points.emplace_back(x,y,z);
  }
  for (std::size_t i=0; i<nf; ++i) {
    std::size_t n; Triangle t;
    if (!(f >> n >> t[0] >> t[1] >> t[2]) || n != 3 ||
        t[0]>=nv || t[1]>=nv || t[2]>=nv) throw std::runtime_error("Invalid triangle");
    soup.triangles.push_back(t);
  }
  return soup;
}

void write(const std::string& path, const Soup& soup) {
  std::ofstream f(path); f << std::setprecision(17);
  f << "OFF\n" << soup.points.size() << " " << soup.triangles.size() << " 0\n";
  for (const auto& p : soup.points)
    f << CGAL::to_double(p.x()) << " " << CGAL::to_double(p.y()) << " " << CGAL::to_double(p.z()) << "\n";
  for (const auto& t : soup.triangles) f << "3 " << t[0] << " " << t[1] << " " << t[2] << "\n";
  if (!f) throw std::runtime_error("Unable to write output");
}

// A disconnected triangle graph accepts non-manifold input soups without
// dropping faces. Distance is to the geometric union of all input triangles.
CGAL::Surface_mesh<Point> graph(const Soup& soup) {
  CGAL::Surface_mesh<Point> mesh;
  for (const auto& t : soup.triangles) {
    auto a=mesh.add_vertex(soup.points[t[0]]);
    auto b=mesh.add_vertex(soup.points[t[1]]);
    auto c=mesh.add_vertex(soup.points[t[2]]);
    mesh.add_face(a,b,c);
  }
  return mesh;
}

// Certify entire triangles by an exact 2D arrangement of nearby projected
// target facets. Every lifted point is within `budget`, including interiors;
// this is a surface coverage bound, not a sampled maximum.
bool covers(const Soup& source, const Soup& target, double budget) {
  using Tri=Kernel::Triangle_3;
  using Triangles=std::vector<Tri>;
  using Primitive=CGAL::AABB_triangle_primitive_3<Kernel,Triangles::iterator>;
  using Traits=CGAL::AABB_traits_3<Kernel,Primitive>;
  Triangles triangles;
  for (const auto& t:target.triangles)
    triangles.emplace_back(target.points[t[0]],target.points[t[1]],target.points[t[2]]);
  CGAL::AABB_tree<Traits> tree(triangles.begin(),triangles.end());
  for (const auto& t:source.triangles) {
    Point a=source.points[t[0]],b=source.points[t[1]],c=source.points[t[2]];
    Kernel::Plane_3 plane(a,b,c);
    std::array<Kernel::FT,3> coefficients{plane.a(),plane.b(),plane.c()};
    int axis=0;
    for (int k=1;k<3;++k)
      if (CGAL::abs(coefficients[k])>CGAL::abs(coefficients[axis])) axis=k;
    if (coefficients[axis]==0) return false;
    auto project=[axis](const Point& p) {
      if(axis==0) return Kernel::Point_2(p.y(),p.z());
      if(axis==1) return Kernel::Point_2(p.x(),p.z());
      return Kernel::Point_2(p.x(),p.y());
    };
    auto polygon=[&](const Tri& tri) {
      CGAL::Polygon_2<Kernel> poly;
      for(int k=0;k<3;++k) poly.push_back(project(tri[k]));
      if(poly.orientation()==CGAL::CLOCKWISE) poly.reverse_orientation();
      return poly;
    };
    CGAL::Polygon_set_2<Kernel> remainder(polygon(Tri(a,b,c)));
    auto bbox=Tri(a,b,c).bbox();
    Kernel::Iso_cuboid_3 query(bbox.xmin()-budget,bbox.ymin()-budget,bbox.zmin()-budget,
                             bbox.xmax()+budget,bbox.ymax()+budget,bbox.zmax()+budget);
    std::vector<Triangles::iterator> hits;
    tree.all_intersected_primitives(query,std::back_inserter(hits));
    for(const auto& hit:hits) {
      bool near=true;
      for(int k=0;k<3;++k) {
        const auto& p=(*hit)[k];
        auto drift=CGAL::abs(plane.a()*p.x()+plane.b()*p.y()+plane.c()*p.z()+plane.d());
        if(drift>Kernel::FT(budget/2)*CGAL::abs(coefficients[axis])) { near=false; break; }
      }
      if(!near) continue;
      auto poly=polygon(*hit);
      if(poly.area()==0) continue;
      // Permit a bounded in-plane numerical envelope as well as plane drift.
      // |dx|,|dy| <= b/16 and |dz| <= b/2+2*b/16 give 3D
      // displacement below b. This closes rounding slivers without accepting
      // a point outside the declared surface-distance budget.
      std::vector<Kernel::Point_2> envelope;
      Kernel::FT radius(budget/16);
      for(const auto& point:poly)
        for(int x:{-1,1}) for(int y:{-1,1})
          envelope.emplace_back(point.x()+x*radius,point.y()+y*radius);
      CGAL::Polygon_2<Kernel> expanded;
      CGAL::convex_hull_2(envelope.begin(),envelope.end(),std::back_inserter(expanded));
      remainder.difference(expanded);
      if(remainder.is_empty()) break;
    }
    if(!remainder.is_empty()) return false;
  }
  return true;
}

Soup clean_coplanar(const Soup& soup) {
  // Delete only complete facets whose surface is exactly covered by retained
  // coplanar intersecting facets. No distance envelope or coordinate movement.
  std::vector<std::pair<std::size_t,std::size_t>> pairs;
  PMP::triangle_soup_self_intersections(soup.points,soup.triangles,std::back_inserter(pairs),
                                        CGAL::parameters::maximum_number(8192));
  std::map<std::size_t,std::vector<std::size_t>> neighbors;
  for(const auto& pair:pairs) {
    if(pair.first==pair.second) continue;
    const auto& a=soup.triangles[pair.first];
    const auto& b=soup.triangles[pair.second];
    Kernel::Plane_3 plane(soup.points[a[0]],soup.points[a[1]],soup.points[a[2]]);
    if(!plane.has_on(soup.points[b[0]]) || !plane.has_on(soup.points[b[1]]) || !plane.has_on(soup.points[b[2]])) continue;
    neighbors[pair.first].push_back(pair.second);
    neighbors[pair.second].push_back(pair.first);
  }
  std::vector<std::size_t> order;
  for(const auto& entry:neighbors) order.push_back(entry.first);
  auto area=[&](std::size_t i) {
    const auto& t=soup.triangles[i];
    return Kernel::Triangle_3(soup.points[t[0]],soup.points[t[1]],soup.points[t[2]]).squared_area();
  };
  std::sort(order.begin(),order.end(),[&](auto a,auto b) {
    auto aa=area(a),ab=area(b);
    return aa==ab ? a<b : aa<ab;
  });
  std::vector<bool> removed(soup.triangles.size(),false);
  std::size_t attempted=0;
  for(auto i:order) {
    if(++attempted>512) break;
    if(neighbors[i].size()>64) continue;
    const auto& face=soup.triangles[i];
    Soup source{{soup.points[face[0]],soup.points[face[1]],soup.points[face[2]]},{{0,1,2}}},target;
    for(auto j:neighbors[i]) if(!removed[j]) {
      const auto& other=soup.triangles[j];
      auto start=target.points.size();
      for(auto v:other) target.points.push_back(soup.points[v]);
      target.triangles.push_back({start,start+1,start+2});
    }
    if(!target.triangles.empty() && covers(source,target,0)) removed[i]=true;
  }
  Soup result{soup.points,{}};
  for(std::size_t i=0;i<soup.triangles.size();++i)
    if(!removed[i]) result.triangles.push_back(soup.triangles[i]);
  if(result.triangles.empty()) throw std::runtime_error("Coplanar cleanup cannot empty a mesh");
  return result;
}

Soup fill(const Soup& soup, std::size_t edge_limit) {
  CGAL::Surface_mesh<Point> mesh;
  std::vector<CGAL::Surface_mesh<Point>::Vertex_index> vertices;
  for(const auto& p:soup.points) vertices.push_back(mesh.add_vertex(p));
  for(const auto& t:soup.triangles)
    if(mesh.add_face(vertices[t[0]],vertices[t[1]],vertices[t[2]])==mesh.null_face())
      throw std::runtime_error("Hole triangulation requires coherent orientable manifold sheets");
  std::vector<CGAL::Surface_mesh<Point>::Halfedge_index> boundaries;
  PMP::extract_boundary_cycles(mesh,std::back_inserter(boundaries));
  for(const auto& h:boundaries) {
    if(!mesh.is_border(h)) continue;
    std::size_t count=0;
    for(auto v:CGAL::vertices_around_face(h,mesh)) { (void)v; ++count; }
    if(count>edge_limit) continue;
    PMP::triangulate_hole(mesh,h);
  }
  Soup result;
  for(auto v:mesh.vertices()) result.points.push_back(mesh.point(v));
  for(auto f:mesh.faces()) {
    Triangle t; int k=0;
    for(auto v:CGAL::vertices_around_face(mesh.halfedge(f),mesh)) t[k++]=v.idx();
    if(k!=3) throw std::runtime_error("Nontriangular hole output");
    result.triangles.push_back(t);
  }
  return result;
}

bool component_intersections(const Soup& soup) {
  std::vector<std::size_t> parent(soup.triangles.size());
  std::iota(parent.begin(),parent.end(),0);
  auto root=[&](std::size_t i) { while(parent[i]!=i) { parent[i]=parent[parent[i]]; i=parent[i]; } return i; };
  std::map<std::pair<std::size_t,std::size_t>,std::size_t> edges;
  for(std::size_t i=0;i<soup.triangles.size();++i) {
    const auto& t=soup.triangles[i];
    for(int k=0;k<3;++k) {
      auto a=t[k],b=t[(k+1)%3]; if(a>b) std::swap(a,b);
      auto inserted=edges.emplace(std::make_pair(a,b),i);
      if(!inserted.second) parent[root(i)]=root(inserted.first->second);
    }
  }
  std::map<std::size_t,std::vector<Triangle>> groups;
  for(std::size_t i=0;i<soup.triangles.size();++i) groups[root(i)].push_back(soup.triangles[i]);
  std::vector<std::vector<Triangle>> components;
  std::vector<bool> negative;
  for(const auto& entry:groups) {
    const auto& triangles=entry.second;
    if(PMP::does_triangle_soup_self_intersect(soup.points,triangles)) return true;
    const Point& origin=soup.points[triangles.front()[0]];
    Kernel::FT volume=0;
    for(const auto& t:triangles) {
      auto a=soup.points[t[0]]-origin,b=soup.points[t[1]]-origin,c=soup.points[t[2]]-origin;
      volume+=a*CGAL::cross_product(b,c);
    }
    components.push_back(triangles); negative.push_back(volume<0);
  }
  // A void or island touching another boundary is not a valid nested material
  // interface. Positive disconnected objects may have zero-volume contacts;
  // their positive-volume overlap is checked separately in Manifold64.
  for(std::size_t i=0;i<components.size();++i)
    for(std::size_t j=i+1;j<components.size();++j) if(negative[i] || negative[j]) {
      auto pair=components[i]; pair.insert(pair.end(),components[j].begin(),components[j].end());
      if(PMP::does_triangle_soup_self_intersect(soup.points,pair)) return true;
    }
  return false;
}

int main(int argc, char** argv) {
  try {
    if (argc<3) throw std::runtime_error("Expected check/refine/distance and an OFF path");
    std::string action=argv[1]; Soup soup=read(argv[2]);
    std::cout << std::setprecision(17);
    if (action=="check_components") {
      bool intersects=component_intersections(soup);
      std::cout << "{\"self_intersection\":" << (intersects ? "true" : "false") << "}\n";
    } else if (action=="check") {
      bool intersects=PMP::does_triangle_soup_self_intersect(soup.points,soup.triangles);
      std::cout << "{\"self_intersection\":" << (intersects ? "true" : "false") << "}\n";
    } else if (action=="intersections") {
      constexpr std::size_t limit=8192;
      std::vector<std::pair<std::size_t,std::size_t>> pairs;
      PMP::triangle_soup_self_intersections(soup.points,soup.triangles,std::back_inserter(pairs),
                                            CGAL::parameters::maximum_number(limit+1));
      std::cout << "{\"complete\":" << (pairs.size()<=limit ? "true" : "false") << ",\"pairs\":[";
      for(std::size_t i=0;i<std::min(limit,pairs.size());++i) {
        if(i) std::cout << ',';
        std::cout << '[' << pairs[i].first << ',' << pairs[i].second << ']';
      }
      std::cout << "]}\n";
    } else if (action=="clean_coplanar" && argc==4) {
      auto result=clean_coplanar(soup);
      write(argv[3],result);
      std::cout << "{\"removed_faces\":" << soup.triangles.size()-result.triangles.size() << "}\n";
    } else if (action=="fill" && argc==5) {
      write(argv[3],fill(soup,std::stoull(argv[4])));
      std::cout << "{\"filled\":true}\n";
    } else if (action=="refine" && argc==4) {
      // Exact constructions first; conversion to doubles is subsequently
      // checked by the independent validator and bidirectional distance gate.
      bool ok=PMP::autorefine_triangle_soup(soup.points,soup.triangles);
      if (!ok) throw std::runtime_error("Autorefinement failed");
      write(argv[3],soup);
      std::cout << "{\"refined\":true}\n";
    } else if (action=="covers" && argc==5) {
      bool covered=covers(soup,read(argv[3]),std::stod(argv[4]));
      std::cout << "{\"covered\":" << (covered ? "true" : "false") << "}\n";
    } else if (action=="distance" && argc==5) {
      auto source=graph(soup); auto target=graph(read(argv[3]));
      double error=std::stod(argv[4]);
      if (!(error>0)) throw std::runtime_error("Distance error bound must be positive");
      double distance=PMP::bounded_error_Hausdorff_distance<CGAL::Sequential_tag>(source,target,error);
      std::cout << "{\"distance\":" << distance << ",\"error_bound\":" << error << "}\n";
    } else throw std::runtime_error("Unknown worker operation");
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
