/*
 * Refactored COMSOL model for wall tomography simulation
 * Compile with command:
 * & "C:\Program Files\COMSOL\COMSOL60\Multiphysics\bin\win64\comsolcompile.exe" ` "C:\Users\Jaime\Documents\deep-learning-wall-tomography\COMSOL\Refactored.java"
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on Jul 8 2025, 13:58 by COMSOL 6.0.0.405. */
public class Refactored {

  	// Entry point: create and configure the COMSOL model
	public static Model run() {
		// Initialize a new model named "Model"
		Model model = ModelUtil.create("Model");

		// Set the working directory for model files
		model.modelPath("C:\\Users\\Jaime\\Documents\\deep-learning-wall-tomography\\COMSOL");

		// Assign a label to the model file
		model.label("New.mph");

		// ----------------------------------------------------------------
		// Define model parameters
		// ----------------------------------------------------------------

		// Signal parameters
		model.param().set("f0", "54 [kHz]", "Signal frequency");
		model.param().set("T0", "1/f0", "Signal period");
		model.param().set("dS", "3e-7 [1/m^2]", "Signal source extent");
		model.param().set("dt", "5e-7 [s]", "Time step");
		model.param().set("tf", "2e-4 [s]", "Time window");

		// Material properties: stone
		model.param().set("E_st", "11 [GPa]", "E modulus stone");
		model.param().set("nu_st", "0.2", "Poisson's ratio stone");
		model.param().set("cp_stone", "2700 [m/s]", "Pressure wave speed, stone");
		model.param().set("cs_stone", "1450 [m/s]", "Shear wave speed, stone");
		model.param().set("rho_st", "1775 [kg/m^3]", "Density stone");

		// Material properties: mortar (plastic)
		model.param().set("E_pl", "2.5 [GPa]", "E modulus mortar");
		model.param().set("nu_pl", "0.35", "Poisson's ratio mortar");
		model.param().set("cp_plastic", "1700 [m/s]", "Pressure wave speed, mortar");
		model.param().set("cs_plastic", "850 [m/s]", "Shear wave speed, mortar");
		model.param().set("rho_pl", "1200 [kg/m^3]", "Density plastic");

		// Signal spread parameter (standard deviation)
		model.param().set("std", "3*T0/4", "sigma value");

		// ----------------------------------------------------------------
		// Define analytic functions for emission
		// ----------------------------------------------------------------

		// Temporal emission function H_time: Gaussian-modulated cosine pulse
		model.func().create("an1", "Analytic");
		model.func("an1").label("emission_time");
		model.func("an1").set("funcname", "H_time");
		model.func("an1").set("expr", "100*exp(-(((t-3*T0)^2)/(2*(std^2))))*cos(2*pi*f0*(t-T0))");
		model.func("an1").set("args", new String[]{"t"});
		model.func("an1").set("argunit", new String[]{"s"});
		model.func("an1").set("plotargs", new String[][]{{"t", "0", "tf"}});

		// Spatial emission function H_space: 2D Gaussian distribution
		model.func().create("an2", "Analytic");
		model.func("an2").label("emission_space");
		model.func("an2").set("funcname", "H_space");
		model.func("an2").set("expr", "1/sqrt(pi*dS)*exp(-(x^2+y^2)/dS)");
		model.func("an2").set("args", new String[]{"x", "y"});
		model.func("an2").set("argunit", new String[]{"m", "m"});
		model.func("an2").set("plotargs", new String[][]{{"x", "-0.01", "0.01"}, {"y", "-0.01", "0.01"}});

		// -----------------------------------------------------------
		// Initialize main model component, its geometry, and mesh
		// -----------------------------------------------------------

		// Create the primary component "comp1" and add a 3D geometry "geom1"
		model.component().create("comp1", true);
		String geom = "geom1";
		model.component("comp1").geom().create(geom, 3);
		

		// Use the COMSOL mesh representation for this geometry
		model.component("comp1").geom(geom).geomRep("comsol");

		// Create the top-level mesh container "mesh1" for comp1
		model.component("comp1").mesh().create("mesh1");

		// ----------------------------------------------------------------
		// Create mesh‐components, import STL’s, and embed into comp1.geom
		// ----------------------------------------------------------------

		String baseDir = "C:\\Users\\Jaime\\Documents\\deep-learning-wall-tomography\\sections_generator\\output3d\\stlsNormal\\00000\\";
		String[] stlFiles = {
			"00000_stone001.stl",
			"00000_stone003.stl",
			"00000_stone004.stl",
			"00000_stone002.stl",
			"00000_stone000.stl"
		};

		for (int i = 1; i <= stlFiles.length; i++) {
			// Tags for this slice
			String compTag  = "mcomp"  + i;
			String geomTag  = "mgeom"  + i;
			String partTag  = "mpart"  + i;
			String impMesh   = "imp1";         // mesh‐import feature name
			String impGeom   = "imp"  + i;     // comp1‐geom import tag
			String filename  = baseDir + stlFiles[i-1];

			// Create mesh component + its geometry
			model.component().create(compTag, "MeshComponent");
			model.geom().create(geomTag, 3);

			// Link geometry into a mesh part
			model.mesh().create(partTag, geomTag);

			// Import the STL into the mesh part and run meshing
			model.mesh(partTag).create(impMesh, "Import");
			model.mesh(partTag).feature(impMesh).set("source",   "stl");
			model.mesh(partTag).feature(impMesh).set("filename", filename);
			model.mesh(partTag).run();

			// Import that mesh into comp1’s geometry
			model.component("comp1").geom(geom).create(impGeom, "Import");
			model.component("comp1").geom(geom).feature(impGeom).set("type", "mesh");
			model.component("comp1").geom(geom).feature(impGeom).set("mesh", partTag);
			model.component("comp1").geom(geom).feature(impGeom).importData();
		}

		// ----------------------------------------------------------------
		// Construct base block and subtract imported geometries
		// ----------------------------------------------------------------
		model.component("comp1").geom(geom).create("blk1", "Block");
		model.component("comp1").geom(geom).feature("blk1").set("size", new double[]{0.06, 0.04, 0.02});

		model.component("comp1").geom(geom).create("dif1", "Difference");
		model.component("comp1").geom(geom).feature("dif1").set("keepsubtract", true);
		model.component("comp1").geom(geom).feature("dif1").selection("input").set("blk1");
		model.component("comp1").geom(geom).feature("dif1").selection("input2").set("imp1", "imp2", "imp3", "imp4", "imp5");

		// ----------------------------------------------------------------
		// Define key points within geometry for later selection
		// ----------------------------------------------------------------
		double[][] points = {{0.005,0,0.01},{0.015,0,0.01},{0.025,0,0.01},{0.035,0,0.01},
						{0.045,0,0.01},{0.055,0,0.01},{0.005,0.04,0.01},{0.01,0.04,0.01},
						{0.015,0.04,0.01},{0.02,0.04,0.01},{0.025,0.04,0.01},{0.03,0.04,0.01},
						{0.035,0.04,0.01},{0.04,0.04,0.01},{0.045,0.04,0.01},{0.05,0.04,0.01},
						{0.055,0.04,0.01}};
		for (int idx = 0; idx < points.length; idx++) {
		String pid = "pt" + (idx+1);
		model.component("comp1").geom(geom).create(pid, "Point");
		model.component("comp1").geom(geom).feature(pid).set("p", points[idx]);
		}

		// ----------------------------------------------------------------
		// Group geometry entities using Unions for assembly
		// ----------------------------------------------------------------

		// Define unions between geometry and points
		model.component("comp1").geom("geom1").create("uni1", "Union");
		model.component("comp1").geom("geom1").feature("uni1").selection("input")
			.set("imp1");

		model.component("comp1").geom("geom1").create("uni2", "Union");
		model.component("comp1").geom("geom1").feature("uni2").selection("input")
			.set("imp2", "pt10", "pt11", "pt12", "pt13");

		model.component("comp1").geom("geom1").create("uni3", "Union");
		model.component("comp1").geom("geom1").feature("uni3").selection("input")
			.set("imp3", "pt16", "pt17");

		model.component("comp1").geom("geom1").create("uni4", "Union");
		model.component("comp1").geom("geom1").feature("uni4").selection("input")
			.set("imp4", "pt6");

		model.component("comp1").geom("geom1").create("uni5", "Union");
		model.component("comp1").geom("geom1").feature("uni5").selection("input")
			.set("imp5", "pt1", "pt7", "pt8");

		model.component("comp1").geom("geom1").create("uniMortar", "Union");
		model.component("comp1").geom("geom1").feature("uniMortar").selection("input")
			.set("dif1", "pt14", "pt15", "pt2", "pt3", "pt4", "pt5", "pt9");

		// Finalize assembly of all geometric parts
		model.component("comp1").geom(geom).feature("fin").label("Form Assembly");
		model.component("comp1").geom(geom).feature("fin").set("action","assembly");
		model.component("comp1").geom(geom).feature("fin").set("imprint",true);

		// Create node group for points
		model.component("comp1").geom(geom).nodeGroup().create("grp1");
		model.component("comp1").geom(geom).nodeGroup("grp1").label("points");
		model.component("comp1").geom(geom).nodeGroup("grp1").placeAfter("dif1");
		for (int i = 1; i <= 17; i++) {
		model.component("comp1").geom(geom).nodeGroup("grp1").add("pt"+i);
		}

		// Create node group for unions
		model.component("comp1").geom("geom1").nodeGroup().create("grp2");
		model.component("comp1").geom("geom1").nodeGroup("grp2").label("unions");
		model.component("comp1").geom("geom1").nodeGroup("grp2").placeAfter("dif1");
		model.component("comp1").geom("geom1").nodeGroup("grp2").add("uni1");
		model.component("comp1").geom("geom1").nodeGroup("grp2").add("uni2");
		model.component("comp1").geom("geom1").nodeGroup("grp2").add("uni3");
		model.component("comp1").geom("geom1").nodeGroup("grp2").add("uni4");
		model.component("comp1").geom("geom1").nodeGroup("grp2").add("uni5");
		model.component("comp1").geom("geom1").nodeGroup("grp2").add("uniMortar");

		// Run the geometry to finalize the model
		model.component("comp1").geom(geom).run();
		model.component("comp1").geom(geom).run("fin");

		// ----------------------------------------------------------------
		// Material definitions and assignments
		// ----------------------------------------------------------------
		// Create and label stone material, assign to relevant domains
		model.component("comp1").material().create("mat1", "Common");  // Stone
		model.component("comp1").material("mat1").label("stone");
		model.component("comp1").material("mat1").selection().set(1, 4, 5, 6, 7);
		// Define elastic properties group and set density, Young's modulus, Poisson's ratio
		model.component("comp1").material("mat1").propertyGroup().create("Enu", "Young's modulus and Poisson's ratio");
		model.component("comp1").material("mat1").propertyGroup("def").set("density", "rho_st");
		model.component("comp1").material("mat1").propertyGroup("Enu").set("E", "E_st");
		model.component("comp1").material("mat1").propertyGroup("Enu").set("nu", "nu_st");

		// Create and label mortar material, assign to relevant domains
		model.component("comp1").material().create("mat2", "Common");  // Mortar
		model.component("comp1").material("mat2").label("mortar");
		model.component("comp1").material("mat2").selection().set(2, 3);
		model.component("comp1").material("mat2").propertyGroup().create("Enu", "Young's modulus and Poisson's ratio");
		model.component("comp1").material("mat2").propertyGroup("def").set("density", "rho_pl");
		model.component("comp1").material("mat2").propertyGroup("Enu").set("E", "E_pl");
		model.component("comp1").material("mat2").propertyGroup("Enu").set("nu", "nu_pl");

		// ----------------------------------------------------------------
		// Physics definitions: Solid Mechanics for source + receivers
		// ----------------------------------------------------------------
		String[] solids = {"solid","solid2","solid3","solid4","solid5","solid6"};
		int[] ptSel   = {9,44,76,87,121,205}; // Emission points

		for (int i = 0; i < solids.length; i++) {
		String tag = solids[i];
		// Create Solid Mechanics physics
		model.component("comp1").physics().create(tag, "SolidMechanics", "geom1");

		// Damping (Rayleigh) around excitation band
		model.component("comp1").physics(tag).feature("lemm1")
			.create("dmp1", "Damping", 3);
		model.component("comp1").physics(tag).feature("lemm1")
			.feature("dmp1").set("InputParameters", "DampingRatios");
		model.component("comp1").physics(tag).feature("lemm1")
			.feature("dmp1").set("f1", "0.99*f0");
		model.component("comp1").physics(tag).feature("lemm1")
			.feature("dmp1").set("zeta1", "5e-3");
		model.component("comp1").physics(tag).feature("lemm1")
			.feature("dmp1").set("f2", "1.01*f0");
		model.component("comp1").physics(tag).feature("lemm1")
			.feature("dmp1").set("zeta2", "5e-3");

		// Point load at emission or receiver location
		model.component("comp1").physics(tag)
			.create("pl1", "PointLoad", 0);
		model.component("comp1").physics(tag).feature("pl1")
			.selection().set(ptSel[i]);
		model.component("comp1").physics(tag).feature("pl1")
			.set("Fp", new String[][]{{"0"},{"H_time(t)"},{"0"}});

		// Roller constraints
		model.component("comp1").physics(tag)
			.create("roll1", "Roller", 2);
		model.component("comp1").physics(tag).feature("roll1")
			.selection().set(5,16,71,109,120,121);
		model.component("comp1").physics(tag)
			.create("roll2", "Roller", 2);
		model.component("comp1").physics(tag).feature("roll2")
			.selection().set(4,19,24,75,88,101,106, 113);

		// PairThinElasticLayer for far-field
		model.component("comp1").physics(tag)
			.create("tel1", "PairThinElasticLayer", 2);
		model.component("comp1").physics(tag).feature("tel1")
			.set("kPerArea", new String[][]{{"1e6"},{"0"},{"0"},{"0"},{"1e6"},{"0"},{"0"},{"0"},{"1e6"}});
		model.component("comp1").physics(tag).feature("tel1")
			.set("mPerArea", 100);
		model.component("comp1").physics(tag).feature("tel1")
			.set("pairs", new String[][]{{"ap1"},{"ap2"},{"ap3"},{"ap4"}, {"ap5"}});

		// Spring foundation for far-field
		model.component("comp1").physics(tag)
			.create("spf1", "SpringFoundation2", 2);
		model.component("comp1").physics(tag).feature("spf1")
			.set("kPerArea", new String[][]{{"1e6"},{"0"},{"0"},{"0"},{"1e6"},{"0"},{"0"},{"0"},{"1e6"}});
		model.component("comp1").physics(tag).feature("spf1")
			.selection().set(3,18,23,74,87,100,112);
		}

		// ----------------------------------------------------------------
		// Studies: time-dependent for each physics set
		// ----------------------------------------------------------------
		String[] stdTags = {"std1","std2","std3","std4","std5","std6"};
		String[][] activates = {
		{"solid","on","solid2","off","solid3","off","solid4","off","solid5","off","solid6","off","frame:spatial1","on","frame:material1","on"},
		{"solid","off","solid2","on","solid3","off","solid4","off","solid5","off","solid6","off","frame:spatial1","on","frame:material1","on"},
		{"solid","off","solid2","off","solid3","on","solid4","off","solid5","off","solid6","off","frame:spatial1","on","frame:material1","on"},
		{"solid","off","solid2","off","solid3","off","solid4","on","solid5","off","solid6","off","frame:spatial1","on","frame:material1","on"},
		{"solid","off","solid2","off","solid3","off","solid4","off","solid5","on","solid6","off","frame:spatial1","on","frame:material1","on"},
		{"solid","off","solid2","off","solid3","off","solid4","off","solid5","off","solid6","on","frame:spatial1","on","frame:material1","on"}
		};
		for (int i = 0; i < stdTags.length; i++) {
			String std = stdTags[i];
			model.study().create(std);
			model.study(std).create("time", "Transient");
			model.study(std).feature("time").set("activate", activates[i]);
			model.study(std).feature("time").set("tlist", "range(0,dt,tf)");
		}

		// ----------------------------------------------------------------
		// Solvers: solver definitions for each study
		// ----------------------------------------------------------------

		// Create solvers for each study
		String[] solTags = {"sol1","sol2","sol3","sol4","sol5","sol6"};
		for (int j = 0; j < solTags.length; j++) {
			String sol = solTags[j];
			String std = "std" + (j+1);

			model.sol().create(sol);
			model.sol(sol).study(std);
			model.sol(sol).attach(std);
			model.sol(sol).create("st1", "StudyStep");
			model.sol(sol).create("v1", "Variables");
			model.sol(sol).create("t1", "Time");
			model.sol(sol).feature("t1").create("fc1", "FullyCoupled");
			model.sol(sol).feature("t1").create("d1", "Direct");
			model.sol(sol).feature("t1").create("i1", "Iterative");
			model.sol(sol).feature("t1").feature("i1").create("mg1", "Multigrid");
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("pr").create("so1", "SOR");
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("po").create("so1", "SOR");
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("cs").create("d1", "Direct");
			// finalize and run
			model.sol(sol).feature("t1").feature().remove("fcDef");
		}

		// Create plots (grouped by 1D and 3D for each study)
		for (int i = 1; i <= 6; i++) {
			int pg1 = 2*i - 1;
			int pg2 = 2*i;
			// Create 1D and 3D groups
			model.result().create("pg" + pg1, "PlotGroup1D");
			model.result().create("pg" + pg2, "PlotGroup3D");
			
			// 1D PointGraph
			model.result("pg" + pg1).create("ptgr1", "PointGraph");
			model.result("pg" + pg1).feature("ptgr1").set("data", "dset" + i);
			model.result("pg" + pg1).feature("ptgr1").selection().set(10,13,45,151,154,155,158,106,122,222,225);
			String varExpr = (i == 1) ? "" : "" + i; // omit index for first dataset
			model.result("pg" + pg1).feature("ptgr1").set("expr", "v" + varExpr);

			// 3D Isosurface
			model.result("pg" + pg2).create("iso1", "Isosurface");
			model.result("pg" + pg2).feature("iso1").set("data", "dset" + i);
			model.result("pg" + pg2).feature("iso1").set("expr", "solid" + varExpr + ".disp");

			model.result().export().create("plot" + i, "Plot");

			// Group by study
			model.nodeGroup().create("grp" + i, "Results");
			model.nodeGroup("grp" + i).placeAfter(null);
			model.nodeGroup("grp" + i).add("plotgroup", "pg" + pg1);
   			model.nodeGroup("grp" + i).add("plotgroup", "pg" + pg2);
		}


		// Solver settings for each study
		String[] compFeatNames = new String[6];
		String[] solidNames    = {"solid","solid2","solid3","solid4","solid5","solid6"};
		for (int i = 0; i < solTags.length; i++) {
			String sol = solTags[i];
			String std = stdTags[i];
			String idx = (i == 0 ? "" : Integer.toString(i+1));

			// attach study and label steps
			model.sol(sol).attach(std);
			model.sol(sol).feature("st1").label("Compile Equations: Time Dependent");
			model.sol(sol).feature("v1").label("Dependent Variables 1.1");
			model.sol(sol).feature("v1")
				.set("clist", new String[]{"range(0,dt,tf)", "5.0E-7[s]"});

			// scaling of u-component
			String compFeat = "comp1_u" + idx;
			model.sol(sol).feature("v1").feature(compFeat)
				.set("scalemethod", "manual")
				.set("scaleval", "1e-2*0.07483314773547906");

			// time-dependent solver settings
			model.sol(sol).feature("t1").label("Time-Dependent Solver 1.1");
			model.sol(sol).feature("t1").set("tlist",       "range(0,dt,tf)");
			model.sol(sol).feature("t1").set("rtol",        0.001);
			model.sol(sol).feature("t1").set("timemethod",  "genalpha");
			model.sol(sol).feature("t1").set("tstepsgenalpha",  "manual");
			model.sol(sol).feature("t1").set("timestepgenalpha", "dt");

			// solver chaining: direct, advanced, fully coupled
			model.sol(sol).feature("t1").feature("dDef").label("Direct 2");
			model.sol(sol).feature("t1").feature("aDef").label("Advanced 1");
			model.sol(sol).feature("t1").feature("aDef").set("cachepattern", true);
			model.sol(sol).feature("t1").feature("fc1").label("Fully Coupled 1.1");
			model.sol(sol).feature("t1").feature("fc1").set("linsolver", "d1");

			// direct solver parameters
			model.sol(sol).feature("t1").feature("d1").label("Suggested Direct Solver (" + solidNames[i] + ")");
			model.sol(sol).feature("t1").feature("d1").set("linsolver",    "pardiso");
			model.sol(sol).feature("t1").feature("d1").set("pivotperturb", 1.0E-9);

			// iterative / multigrid solver parameters
			model.sol(sol).feature("t1").feature("i1").label("Suggested Iterative Solver (" + solidNames[i] + ")");
			model.sol(sol).feature("t1").feature("i1").feature("ilDef")
				.label("Incomplete LU 1");
			model.sol(sol).feature("t1").feature("i1").feature("mg1")
				.label("Multigrid 1.1");
			// presmoother
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("pr")
				.label("Presmoother 1");
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("pr").feature("soDef")
				.label("SOR 2");
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("pr").feature("so1").label("SOR 1.1");
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("pr").feature("so1").set("relax", 0.8);
			// postsmoother
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("po")
				.label("Postsmoother 1");
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("po").feature("soDef")
				.label("SOR 2");
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("po").feature("so1").label("SOR 1.1");
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("po").feature("so1").set("relax", 0.8);
			// coarse solver
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("cs")
				.label("Coarse Solver 1");
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("cs").feature("dDef")
				.label("Direct 2");
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("cs").feature("d1").label("Direct 1.1");
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("cs").feature("d1").set("linsolver",    "pardiso");
			model.sol(sol).feature("t1").feature("i1").feature("mg1").feature("cs").feature("d1").set("pivotperturb", 1.0E-9);

			// run this solver
			model.sol(sol).runAll();
		}

		// Customize plot groups
		for (int i = 1; i <= 6; i++) {
			String pg1 = "pg" + (2*i - 1);
			String pg2 = "pg" + (2*i);
			// Set plot group labels
			model.result(pg1).label("Point Graph Study " + i);
			model.result(pg2).label("Isosurface Study " + i);
			// Set plot group titles
			model.result(pg1).set("title", "Point Graph for Study " + i);
			model.result(pg2).set("title", "Isosurface for Study " + i);
			// Set plot group labels
			model.result(pg1).set("xlabel", "Time [s]");
			model.result(pg1).set("ylabel", "Displacement [m]");
			// Other stuff
			model.result(pg1).set("xlabelactive", false);
			model.result(pg1).set("ylabelactive", false);;
			model.result(pg1).feature("ptgr1")
				.set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x coordinate"}, 
				{"solid.refpnty", "0", "Reference point for moment computation, y coordinate"}, 
				{"solid.refpntz", "0", "Reference point for moment computation, z coordinate"}, 
				{"solid2.refpntx", "0", "Reference point for moment computation, x coordinate"}, 
				{"solid2.refpnty", "0", "Reference point for moment computation, y coordinate"}, 
				{"solid2.refpntz", "0", "Reference point for moment computation, z coordinate"}, 
				{"solid3.refpntx", "0", "Reference point for moment computation, x coordinate"}, 
				{"solid3.refpnty", "0", "Reference point for moment computation, y coordinate"}, 
				{"solid3.refpntz", "0", "Reference point for moment computation, z coordinate"}, 
				{"solid4.refpntx", "0", "Reference point for moment computation, x coordinate"}, 
				{"solid4.refpnty", "0", "Reference point for moment computation, y coordinate"}, 
				{"solid4.refpntz", "0", "Reference point for moment computation, z coordinate"}, 
				{"solid5.refpntx", "0", "Reference point for moment computation, x coordinate"}, 
				{"solid5.refpnty", "0", "Reference point for moment computation, y coordinate"}, 
				{"solid5.refpntz", "0", "Reference point for moment computation, z coordinate"}, 
				{"solid6.refpntx", "0", "Reference point for moment computation, x coordinate"}, 
				{"solid6.refpnty", "0", "Reference point for moment computation, y coordinate"}, 
				{"solid6.refpntz", "0", "Reference point for moment computation, z coordinate"}});
		
			model.result(pg2).feature("iso1").set("looplevel", new int[]{71});
			model.result(pg2).feature("iso1")
				.set("const", new String[][]{{"solid.refpntx", "0", "Reference point for moment computation, x coordinate"}, 
				{"solid.refpnty", "0", "Reference point for moment computation, y coordinate"}, 
				{"solid.refpntz", "0", "Reference point for moment computation, z coordinate"}, 
				{"solid2.refpntx", "0", "Reference point for moment computation, x coordinate"}, 
				{"solid2.refpnty", "0", "Reference point for moment computation, y coordinate"}, 
				{"solid2.refpntz", "0", "Reference point for moment computation, z coordinate"}, 
				{"solid3.refpntx", "0", "Reference point for moment computation, x coordinate"}, 
				{"solid3.refpnty", "0", "Reference point for moment computation, y coordinate"}, 
				{"solid3.refpntz", "0", "Reference point for moment computation, z coordinate"}, 
				{"solid4.refpntx", "0", "Reference point for moment computation, x coordinate"}, 
				{"solid4.refpnty", "0", "Reference point for moment computation, y coordinate"}, 
				{"solid4.refpntz", "0", "Reference point for moment computation, z coordinate"}, 
				{"solid5.refpntx", "0", "Reference point for moment computation, x coordinate"}, 
				{"solid5.refpnty", "0", "Reference point for moment computation, y coordinate"}, 
				{"solid5.refpntz", "0", "Reference point for moment computation, z coordinate"}, 
				{"solid6.refpntx", "0", "Reference point for moment computation, x coordinate"}, 
				{"solid6.refpnty", "0", "Reference point for moment computation, y coordinate"}, 
				{"solid6.refpntz", "0", "Reference point for moment computation, z coordinate"}});
			model.result(pg2).feature("iso1").set("levelmethod", "levels");
			model.result(pg2).feature("iso1").set("levels", "range(0,5e-9,2.0e-6)");
			model.result(pg2).feature("iso1").set("resolution", "normal");
		}

		return model;
	}

	public static void main(String[] args) {
		run();
	}
}