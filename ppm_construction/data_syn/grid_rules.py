import os
import json 
import numpy as np
import random
np.random.seed(42)
random.seed(42)

                        
(
    TYPE_SHORT,
    TYPE_VOLTAGE_SOURCE,
    TYPE_CURRENT_SOURCE,
    TYPE_RESISTOR,
    TYPE_CAPACITOR,
    TYPE_INDUCTOR,

    TYPE_OPEN,               
    TYPE_VCCS,                                                   
    TYPE_VCVS,                                                   
    TYPE_CCCS,                                                   
    TYPE_CCVS,                                                   
    
                           
    TYPE_OPAMP_INVERTING,                         
    TYPE_OPAMP_NONINVERTING,                            
    TYPE_OPAMP_BUFFER,                                
    TYPE_OPAMP_INTEGRATOR,               
    TYPE_OPAMP_DIFFERENTIATOR,                 
    TYPE_OPAMP_SUMMING,                         
    
                                    
    TYPE_BJT_SMALL_SIGNAL,                                      
    TYPE_MOSFET_SMALL_SIGNAL,                            
) = tuple( range(19) )
NUM_NORMAL=6

                            
(
    MEAS_TYPE_NONE,
    MEAS_TYPE_VOLTAGE,
    MEAS_TYPE_CURRENT,
) = tuple( range(3) )

                     
(
    UNIT_MODE_1,
    UNIT_MODE_k,
    UNIT_MODE_m,
    UNIT_MODE_u,
    UNIT_MODE_n,
    UNIT_MODE_p,
) = tuple( range(6) )

                        
vlt7_latex_template = r"""\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{circuitikz}
\begin{document}
\begin{center}
\begin{circuitikz}[line width=1pt]
\ctikzset{tripoles/en amp/input height=0.5};
\ctikzset{inductors/scale=1.2, inductor=american}
<main>
\end{circuitikz}
\end{center}
\end{document}"""
v8_latex_template = r"""\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{circuitikz}
\tikzset{every node/.style={font=<font>}}
\tikzset{every draw/.style={font=<font>}}
\begin{document}
\begin{center}
\begin{circuitikz}[line width=1pt, american]
\ctikzset{tripoles/en amp/input height=0.5};
\ctikzset{inductors/scale=1.2, inductor=american}
\ctikzset{resistors/scale=0.8, resistor=american}
\ctikzset{capacitors/scale=0.8}
<main>
\end{circuitikz}
\end{center}
\end{document}"""
v8_latex_template = r"""\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{circuitikz}
\tikzset{every node/.style={font=<font>}}
\tikzset{every draw/.style={font=<font>}}
\begin{document}
\begin{center}
\begin{circuitikz}[line width=1pt, american]
\ctikzset{tripoles/en amp/input height=0.5};
\ctikzset{inductors/scale=1.2, inductor=american}
\ctikzset{resistors/scale=0.8, resistor=american}
\ctikzset{capacitors/scale=0.8}
<main>
\end{circuitikz}
\end{center}
\end{document}"""

LATEX_TEMPLATES = {
    "v<=7": vlt7_latex_template,
    "v8": v8_latex_template,
    "v9": v8_latex_template,
    "v10": v8_latex_template,
    "v11": v8_latex_template,
}

unit_scales = ["", "k", "m", "\\mu", "n", "p"]

LABEL_TYPE_NUMBER, LABEL_TYPE_STRING = tuple(range(2))                                             
components_latex_info = [("short", "", ""), ("V","V","V"), ("I","I","A"), ("R","R",r"\Omega"), ("C","C","F"), ("L","L","H"),
                         ("open", "", ""), ("cisource", "", ""), ("cvsource", "", ""), ("cisource", "", ""), ("cvsource", "", ""),
                                                
                         ("op amp", "A", ""), ("op amp", "A", ""), ("op amp", "A", ""), 
                         ("op amp", "A", ""), ("op amp", "A", ""), ("op amp", "A", ""),
                                                         
                         ("bjt_small", "Q", "mS"), ("mosfet_small", "M", "mS")]                    

CUR_MODE_1, CUR_MODE_2, CUR_MODE_3, CUR_MODE_4, CUR_MODE_5, CUR_MODE_6 = tuple(range(6))
flow_direction = ["^>", ">_", "^>", "_>"]

def draw_integrator_template(center_x, center_y, resistor_value, capacitor_value, 
                           use_value_annotation=True, label_subscript=1, 
                           orientation="horizontal_lr"):
               
    if use_value_annotation:
        r_label = f"{int(resistor_value)} \\mathrm{{\\Omega}}"
        c_label = f"{int(capacitor_value)} \\mathrm{{F}}"
    else:
        r_label = f"Rint{int(label_subscript)}"
        c_label = f"Cint{int(label_subscript)}"
                                 
    latex_code = f"% Integrator template ({orientation}) centered at ({center_x:.1f}, {center_y:.1f})\n"
                         
    if use_value_annotation:
        tf_label = f"-\\frac{{1}}{{{int(resistor_value)} \\cdot {int(capacitor_value)} \\cdot s}}"
    else:
        tf_label = f"-\\frac{{1}}{{Rint{int(label_subscript)} \\cdot Cint{int(label_subscript)} \\cdot s}}"
    
    if orientation == "horizontal_lr":
                                                                                    
        opamp_x = center_x
        opamp_y = center_y
        
                                               
        input_node_x = opamp_x - 3.0
        input_node_y = opamp_y + 0.5                                             
        
                                          
        output_node_x = opamp_x + 3.0
        output_node_y = opamp_y
        
                                                
        inverting_node_x = opamp_x - 0.8
        inverting_node_y = opamp_y + 0.2
        
                                   
        feedback_top_x = opamp_x + 3.0
        feedback_top_y = opamp_y + 1.5
        feedback_left_x = opamp_x - 1.2
        feedback_left_y = opamp_y + 1.5
        
                            
        latex_code += f"\\draw ({opamp_x:.1f},{opamp_y:.1f}) node[op amp] (opamp{int(label_subscript)}) {{}};\n"
        
                                                                        
        latex_code += f"\\draw ({input_node_x:.1f},{input_node_y:.1f}) to[R, l=${r_label}$] (opamp{int(label_subscript)}.-);\n"
        
                                         
        latex_code += f"\\draw (opamp{int(label_subscript)}.out) to[short] ({output_node_x:.1f},{output_node_y:.1f});\n"
        latex_code += f"\\draw ({output_node_x:.1f},{output_node_y:.1f}) to[short] ({feedback_top_x:.1f},{feedback_top_y:.1f});\n"
        latex_code += f"\\draw ({feedback_top_x:.1f},{feedback_top_y:.1f}) to[C, l=${c_label}$] ({feedback_left_x:.1f},{feedback_left_y:.1f});\n"
        latex_code += f"\\draw ({feedback_left_x:.1f},{feedback_left_y:.1f}) to[short] (opamp{int(label_subscript)}.-);\n"
        
                                           
        latex_code += f"\\draw (opamp{int(label_subscript)}.+) to[short] ++(0,-0.8) node[ground] {{}};\n"
        
                                        
                                                                                                              
        
                                                                                       
        inv_node_num = int(label_subscript) + 30
        latex_code += f"\\node[circle, draw=blue, fill=white, inner sep=1.5pt] at ({inverting_node_x:.1f},{inverting_node_y:.1f}) {{\\textcolor{{blue}}{{\\tiny {inv_node_num}}}}};\n"
        
        input_point = (input_node_x, input_node_y)
        output_point = (output_node_x, output_node_y)
        
    elif orientation == "horizontal_rl":
                                                                                    
        opamp_x = center_x
        opamp_y = center_y
        
                                                
        input_node_x = opamp_x + 3.0
        input_node_y = opamp_y + 0.5                                             
        
                                         
        output_node_x = opamp_x - 3.0
        output_node_y = opamp_y
        
                                                
        inverting_node_x = opamp_x + 0.8
        inverting_node_y = opamp_y + 0.2
        
                                              
        feedback_top_x = opamp_x - 3.0
        feedback_top_y = opamp_y + 1.5
        feedback_right_x = opamp_x + 1.2
        feedback_right_y = opamp_y + 1.5
        
                                                   
        latex_code += f"\\draw ({opamp_x:.1f},{opamp_y:.1f}) node[op amp, xscale=-1] (opamp{int(label_subscript)}) {{}};\n"
        
                                                                        
        latex_code += f"\\draw ({input_node_x:.1f},{input_node_y:.1f}) to[R, l=${r_label}$] (opamp{int(label_subscript)}.-);\n"
        
                                         
        latex_code += f"\\draw (opamp{int(label_subscript)}.out) to[short] ({output_node_x:.1f},{output_node_y:.1f});\n"
        latex_code += f"\\draw ({output_node_x:.1f},{output_node_y:.1f}) to[short] ({feedback_top_x:.1f},{feedback_top_y:.1f});\n"
        latex_code += f"\\draw ({feedback_top_x:.1f},{feedback_top_y:.1f}) to[C, l=${c_label}$] ({feedback_right_x:.1f},{feedback_right_y:.1f});\n"
        latex_code += f"\\draw ({feedback_right_x:.1f},{feedback_right_y:.1f}) to[short] (opamp{int(label_subscript)}.-);\n"
        
                                                                                                              
        latex_code += f"\\draw (opamp{int(label_subscript)}.+) to[short] ++(0,-0.8) node[ground] {{}};\n"
        
                                        
                                                                                                              
        
                                                                                       
        inv_node_num = int(label_subscript) + 30
        latex_code += f"\\node[circle, draw=blue, fill=white, inner sep=1.5pt] at ({inverting_node_x:.1f},{inverting_node_y:.1f}) {{\\textcolor{{blue}}{{\\tiny {inv_node_num}}}}};\n"
        
        input_point = (input_node_x, input_node_y)
        output_point = (output_node_x, output_node_y)
        
    elif orientation == "vertical_bt":
                                                                                  
        opamp_x = center_x
        opamp_y = center_y
        
                                            
        input_node_x = opamp_x - 0.5                                                
        input_node_y = opamp_y - 3.0
        
                                 
        output_node_x = opamp_x
        output_node_y = opamp_y + 3.0
        
                                                
        inverting_node_x = opamp_x - 0.2
        inverting_node_y = opamp_y - 0.8
        
                                     
        feedback_right_x = opamp_x - 1.5
        feedback_right_y = opamp_y + 3.0
        feedback_bottom_x = opamp_x - 1.5
        feedback_bottom_y = opamp_y - 1.2
        
                                                 
        latex_code += f"\\draw ({opamp_x:.1f},{opamp_y:.1f}) node[op amp, rotate=90] (opamp{int(label_subscript)}) {{}};\n"
        
                                                                        
        latex_code += f"\\draw ({input_node_x:.1f},{input_node_y:.1f}) to[R, l=${r_label}$] (opamp{int(label_subscript)}.-);\n"
        
                                         
        latex_code += f"\\draw (opamp{int(label_subscript)}.out) to[short] ({output_node_x:.1f},{output_node_y:.1f});\n"
        latex_code += f"\\draw ({output_node_x:.1f},{output_node_y:.1f}) to[short] ({feedback_right_x:.1f},{feedback_right_y:.1f});\n"
        latex_code += f"\\draw ({feedback_right_x:.1f},{feedback_right_y:.1f}) to[C, l=${c_label}$] ({feedback_bottom_x:.1f},{feedback_bottom_y:.1f});\n"
        latex_code += f"\\draw ({feedback_bottom_x:.1f},{feedback_bottom_y:.1f}) to[short] (opamp{int(label_subscript)}.-);\n"
        
                                             
        latex_code += f"\\draw (opamp{int(label_subscript)}.+) to[short] ++(+0.8,0) node[ground] {{}};\n"
        
                                        
                                                                                                            
        
                                                                                       
        inv_node_num = int(label_subscript) + 30
        latex_code += f"\\node[circle, draw=blue, fill=white, inner sep=2pt] at ({inverting_node_x:.1f},{inverting_node_y:.1f}) {{\\textcolor{{blue}}{{\\tiny {inv_node_num}}}}};\n"
        
        input_point = (input_node_x, input_node_y)
        output_point = (output_node_x, output_node_y)
        
    else:                                
                                                                                  
        opamp_x = center_x
        opamp_y = center_y
        
                                         
        input_node_x = opamp_x + 0.5                                                
        input_node_y = opamp_y + 3.0
        
                                    
        output_node_x = opamp_x
        output_node_y = opamp_y - 3.0
        
                                                
        inverting_node_x = opamp_x + 0.2
        inverting_node_y = opamp_y + 0.8
        
                                                        
        feedback_right_x = opamp_x + 1.5
        feedback_right_y = opamp_y - 3.0
        feedback_top_x = opamp_x + 1.5
        feedback_top_y = opamp_y + 1.0
        
                                                  
        latex_code += f"\\draw ({opamp_x:.1f},{opamp_y:.1f}) node[op amp, rotate=-90] (opamp{int(label_subscript)}) {{}};\n"
        
                                                                        
        latex_code += f"\\draw ({input_node_x:.1f},{input_node_y:.1f}) to[R, l=${r_label}$] (opamp{int(label_subscript)}.-);\n"
        
                                         
        latex_code += f"\\draw (opamp{int(label_subscript)}.out) to[short] ({output_node_x:.1f},{output_node_y:.1f});\n"
        latex_code += f"\\draw ({output_node_x:.1f},{output_node_y:.1f}) to[short] ({feedback_right_x:.1f},{feedback_right_y:.1f});\n"
        latex_code += f"\\draw ({feedback_right_x:.1f},{feedback_right_y:.1f}) to[C, l=${c_label}$] ({feedback_top_x:.1f},{feedback_top_y:.1f});\n"
        latex_code += f"\\draw ({feedback_top_x:.1f},{feedback_top_y:.1f}) to[short] (opamp{int(label_subscript)}.-);\n"
        
                                             
        latex_code += f"\\draw (opamp{int(label_subscript)}.+) to[short] ++(-0.8,0) node[ground] {{}};\n"
        
                                        
                                                                                                             
        
                                                                                       
        inv_node_num = int(label_subscript) + 30
        latex_code += f"\\node[circle, draw=blue, fill=white, inner sep=2pt] at ({inverting_node_x:.1f},{inverting_node_y:.1f}) {{\\textcolor{{blue}}{{\\tiny {inv_node_num}}}}};\n"
        
        input_point = (input_node_x, input_node_y)
        output_point = (output_node_x, output_node_y)
    
                                       
    latex_code += f"\\node[circle, fill=red, inner sep=1.5pt] at ({input_point[0]:.1f},{input_point[1]:.1f}) {{}};\n"                          
    latex_code += f"\\node[circle, fill=red, inner sep=1.5pt] at ({output_point[0]:.1f},{output_point[1]:.1f}) {{}};\n"                           
    
                                                 
    if orientation == "horizontal_lr":
        ground_point = (opamp_x, opamp_y - 1.0)                    
    elif orientation == "horizontal_rl":
        ground_point = (opamp_x, opamp_y + 1.0)                                       
    elif orientation == "vertical_bt":
        ground_point = (opamp_x - 1.0, opamp_y)                    
    else:               
        ground_point = (opamp_x + 1.0, opamp_y)                     
    
    return latex_code, input_point, output_point, ground_point


def draw_bjt_small_signal_template(center_x, center_y, gm_value, rpi_value, ro_value, 
                                  use_value_annotation=True, label_subscript=1, 
                                  orientation="horizontal_lr"):
                  
    if use_value_annotation:
        gm_label = f"{gm_value:.1f} \\mathrm{{mS}}"
        rpi_label = f"{int(rpi_value)} \\mathrm{{\\Omega}}"
        ro_label = f"{int(ro_value)} \\mathrm{{\\Omega}}"
    else:
        gm_label = f"g_{{m{int(label_subscript)}}}"
        rpi_label = f"r_{{\\pi{int(label_subscript)}}}"
        ro_label = f"r_{{o{int(label_subscript)}}}"
    
    latex_code = f"% BJT Small Signal Model ({orientation}) centered at ({center_x:.1f}, {center_y:.1f})\n"
    
    if orientation == "horizontal_lr":
                              
        base_x = center_x - 2.5
        base_y = center_y
        
                                        
        collector_x = center_x + 2.5
        collector_y = center_y + 1.0
        
                                          
        emitter_x = center_x + 2.5
        emitter_y = center_y - 1.0
        
                                 
        internal_x = center_x
        internal_y = center_y
        
                                                       
        latex_code += f"\\draw ({base_x:.1f},{base_y:.1f}) to[R, l=${rpi_label}$] ({internal_x:.1f},{emitter_y:.1f});\n"
        
                                                                         
        latex_code += f"\\draw ({internal_x:.1f},{collector_y:.1f}) to[american current source, l=${gm_label}v_{{be}}$] ({internal_x:.1f},{emitter_y:.1f});\n"
        
                                                                        
        latex_code += f"\\draw ({internal_x + 0.8:.1f},{collector_y:.1f}) to[R, l=${ro_label}$] ({internal_x + 0.8:.1f},{emitter_y:.1f});\n"
        
                                          
        latex_code += f"\\draw ({internal_x:.1f},{collector_y:.1f}) to[short] ({collector_x:.1f},{collector_y:.1f});\n"
        latex_code += f"\\draw ({internal_x:.1f},{emitter_y:.1f}) to[short] ({emitter_x:.1f},{emitter_y:.1f});\n"
        latex_code += f"\\draw ({internal_x + 0.8:.1f},{collector_y:.1f}) to[short] ({collector_x:.1f},{collector_y:.1f});\n"
        latex_code += f"\\draw ({internal_x + 0.8:.1f},{emitter_y:.1f}) to[short] ({emitter_x:.1f},{emitter_y:.1f});\n"
        
                                
        latex_code += f"\\node[left] at ({base_x:.1f},{base_y:.1f}) {{B}};\n"
        latex_code += f"\\node[above] at ({collector_x:.1f},{collector_y:.1f}) {{C}};\n"
        latex_code += f"\\node[below] at ({emitter_x:.1f},{emitter_y:.1f}) {{E}};\n"
        
        base_point = (base_x, base_y)
        collector_point = (collector_x, collector_y)
        emitter_point = (emitter_x, emitter_y)
        
    else:
                                                                      
        base_point = (center_x - 2.5, center_y)
        collector_point = (center_x + 2.5, center_y + 1.0)
        emitter_point = (center_x + 2.5, center_y - 1.0)
    
    return latex_code, base_point, collector_point, emitter_point


def draw_mosfet_small_signal_template(center_x, center_y, gm_value, cgs_value, cgd_value, ro_value,
                                     use_value_annotation=True, label_subscript=1, 
                                     orientation="horizontal_lr"):
                
    if use_value_annotation:
        gm_label = f"{gm_value:.1f} \\mathrm{{mS}}"
        cgs_label = f"{cgs_value:.1f} \\mathrm{{pF}}"
        cgd_label = f"{cgd_value:.1f} \\mathrm{{pF}}"
        ro_label = f"{int(ro_value)} \\mathrm{{\\Omega}}"
    else:
        gm_label = f"g_{{m{int(label_subscript)}}}"
        cgs_label = f"C_{{gs{int(label_subscript)}}}"
        cgd_label = f"C_{{gd{int(label_subscript)}}}"
        ro_label = f"r_{{o{int(label_subscript)}}}"
    
    latex_code = f"% MOSFET Small Signal Model ({orientation}) centered at ({center_x:.1f}, {center_y:.1f})\n"
    
    if orientation == "horizontal_lr":
                              
        gate_x = center_x - 2.5
        gate_y = center_y
        
                                    
        drain_x = center_x + 2.5
        drain_y = center_y + 1.0
        
                                        
        source_x = center_x + 2.5 
        source_y = center_y - 1.0
        
                                 
        internal_gate_x = center_x - 0.5
        internal_drain_x = center_x + 0.5
        internal_source_y = center_y - 1.0
        internal_drain_y = center_y + 1.0
        
                                             
        latex_code += f"\\draw ({gate_x:.1f},{gate_y:.1f}) to[short] ({internal_gate_x:.1f},{gate_y:.1f});\n"
        latex_code += f"\\draw ({internal_gate_x:.1f},{gate_y:.1f}) to[C, l=${cgs_label}$] ({internal_gate_x:.1f},{internal_source_y:.1f});\n"
        latex_code += f"\\draw ({internal_gate_x:.1f},{internal_source_y:.1f}) to[short] ({source_x:.1f},{source_y:.1f});\n"
        
                                            
        latex_code += f"\\draw ({internal_gate_x:.1f},{gate_y:.1f}) to[C, l=${cgd_label}$] ({internal_drain_x:.1f},{internal_drain_y:.1f});\n"
        
                                                                    
        latex_code += f"\\draw ({internal_drain_x:.1f},{internal_drain_y:.1f}) to[american current source, l=${gm_label}v_{{gs}}$] ({internal_drain_x:.1f},{internal_source_y:.1f});\n"
        
                                                                        
        latex_code += f"\\draw ({internal_drain_x + 0.6:.1f},{internal_drain_y:.1f}) to[R, l=${ro_label}$] ({internal_drain_x + 0.6:.1f},{internal_source_y:.1f});\n"
        
                                          
        latex_code += f"\\draw ({internal_drain_x:.1f},{internal_drain_y:.1f}) to[short] ({drain_x:.1f},{drain_y:.1f});\n"
        latex_code += f"\\draw ({internal_drain_x + 0.6:.1f},{internal_drain_y:.1f}) to[short] ({drain_x:.1f},{drain_y:.1f});\n"
        latex_code += f"\\draw ({internal_drain_x:.1f},{internal_source_y:.1f}) to[short] ({source_x:.1f},{source_y:.1f});\n"
        latex_code += f"\\draw ({internal_drain_x + 0.6:.1f},{internal_source_y:.1f}) to[short] ({source_x:.1f},{source_y:.1f});\n"
        
                                
        latex_code += f"\\node[left] at ({gate_x:.1f},{gate_y:.1f}) {{G}};\n"
        latex_code += f"\\node[above] at ({drain_x:.1f},{drain_y:.1f}) {{D}};\n"
        latex_code += f"\\node[below] at ({source_x:.1f},{source_y:.1f}) {{S}};\n"
        
        gate_point = (gate_x, gate_y)
        drain_point = (drain_x, drain_y)
        source_point = (source_x, source_y)
        
    else:
                                                                      
        gate_point = (center_x - 2.5, center_y)
        drain_point = (center_x + 2.5, center_y + 1.0)
        source_point = (center_x + 2.5, center_y - 1.0)
    
    return latex_code, gate_point, drain_point, source_point


def draw_bjt_small_signal_2terminal(center_x, center_y, gm_value, rpi_value, ro_value, 
                                   use_value_annotation=True, label_subscript=1, 
                                   orientation="horizontal_lr"):
                
    if use_value_annotation:
        gm_label = f"{gm_value:.1f} \\mathrm{{mS}}"
        rpi_label = f"{int(rpi_value)} \\mathrm{{\\Omega}}"
        ro_label = f"{int(ro_value)} \\mathrm{{\\Omega}}"
    else:
        gm_label = f"g_{{m{int(label_subscript)}}}"
        rpi_label = f"r_{{\\pi{int(label_subscript)}}}"
        ro_label = f"r_{{o{int(label_subscript)}}}"
    
    latex_code = f"% BJT Small Signal 2-Terminal Model ({orientation}) centered at ({center_x:.1f}, {center_y:.1f})\n"
    
    if orientation == "horizontal_lr":
                                      
        input_x = center_x - 1.5
        input_y = center_y
        
                                               
        output_x = center_x + 1.5
        output_y = center_y
        
                                   
        ground_x = center_x
        ground_y = center_y - 1.0
        
                                         
                                                          
        latex_code += f"\\draw ({input_x:.1f},{input_y:.1f}) to[R, l=${rpi_label}$] ({center_x:.1f},{ground_y:.1f});\n"
        
                                                           
        latex_code += f"\\draw ({center_x:.1f},{output_y:.1f}) to[american current source, l=${gm_label}v_{{be}}$] ({center_x:.1f},{ground_y:.1f});\n"
        
                                          
        latex_code += f"\\draw ({center_x + 0.5:.1f},{output_y:.1f}) to[R, l=${ro_label}$] ({center_x + 0.5:.1f},{ground_y:.1f});\n"
        
                                          
        latex_code += f"\\draw ({center_x:.1f},{output_y:.1f}) to[short] ({output_x:.1f},{output_y:.1f});\n"
        latex_code += f"\\draw ({center_x + 0.5:.1f},{output_y:.1f}) to[short] ({output_x:.1f},{output_y:.1f});\n"
        
                       
        latex_code += f"\\node[left] at ({input_x:.1f},{input_y:.1f}) {{B}};\n"
        latex_code += f"\\node[right] at ({output_x:.1f},{output_y:.1f}) {{C}};\n"
        latex_code += f"\\node[below] at ({center_x:.1f},{ground_y:.1f}) {{E(gnd)}};\n"
        
        input_point = (input_x, input_y)
        output_point = (output_x, output_y)
        
    elif orientation == "vertical_bt" or orientation == "vertical_tb":
                                        
        input_x = center_x
        input_y = center_y - 1.5
        
                                           
        output_x = center_x  
        output_y = center_y + 1.5
        
                                   
        ground_x = center_x - 1.0
        ground_y = center_y
        
                                         
        latex_code += f"\\draw ({input_x:.1f},{input_y:.1f}) to[R, l=${rpi_label}$] ({ground_x:.1f},{center_y:.1f});\n"
        latex_code += f"\\draw ({output_x:.1f},{center_y:.1f}) to[american current source, l=${gm_label}v_{{be}}$] ({ground_x:.1f},{center_y:.1f});\n"
        latex_code += f"\\draw ({output_x:.1f},{center_y + 0.5:.1f}) to[R, l=${ro_label}$] ({ground_x:.1f},{center_y + 0.5:.1f});\n"
        latex_code += f"\\draw ({output_x:.1f},{center_y:.1f}) to[short] ({output_x:.1f},{output_y:.1f});\n"
        latex_code += f"\\draw ({output_x:.1f},{center_y + 0.5:.1f}) to[short] ({output_x:.1f},{output_y:.1f});\n"
        
        latex_code += f"\\node[below] at ({input_x:.1f},{input_y:.1f}) {{B}};\n"
        latex_code += f"\\node[above] at ({output_x:.1f},{output_y:.1f}) {{C}};\n"
        latex_code += f"\\node[left] at ({ground_x:.1f},{center_y:.1f}) {{E(gnd)}};\n"
        
        input_point = (input_x, input_y)
        output_point = (output_x, output_y)
        
    else:
                                                              
        input_point = (center_x - 1.5, center_y)
        output_point = (center_x + 1.5, center_y)
    
    return latex_code, input_point, output_point


def get_latex_line_draw(x1, y1, x2, y2,
                        type_number, 
                        label_subscript,
                        value, 
                        value_unit,
                        use_value_annotation,                                                                             
                        style="chinese",
                        measure_type=MEAS_TYPE_NONE,
                        measure_label="",
                        measure_direction=0,
                        control_label="",
                        label_subscript_type=LABEL_TYPE_NUMBER,
                        direction=0,
                        note='v5',
                        analysis_type="dc_analysis"                               
                    ) -> str:
    
    if direction == 1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    meas_comp_same_direction = (direction == measure_direction)
    
                                                                  
    line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    is_short_line = line_length < 2.5
    is_vertical = abs(x2-x1) < 0.1
    is_horizontal = abs(y2-y1) < 0.1
    
                                                                               
    base_offset = 0.5 if not is_short_line else 0.4
    node_offset = 0.8 if not is_short_line else 0.6
    arrow_offset = 0.4 if not is_short_line else 0.3
    
                                                                     
    controlled_source_offset_multiplier = 1.3 if type_number in [TYPE_VCCS, TYPE_VCVS, TYPE_CCCS, TYPE_CCVS] else 1.0
    measurement_offset_multiplier = 1.2 if measure_type != MEAS_TYPE_NONE else 1.0
    
                                                                                             
    if type_number in [TYPE_VCCS, TYPE_VCVS, TYPE_CCCS, TYPE_CCVS] and measure_type != MEAS_TYPE_NONE:
        base_offset *= 1.5
        node_offset *= 1.7
        arrow_offset *= 1.3
    
    if style == "chinese":
        print(f"drawing between ({x1:.1f},{y1:.1f}) and ({x2:.1f},{y2:.1f}), length={line_length:.2f}\n")
        print(f"type_num: {type_number}, label_num: {label_subscript}, value: {value}, use_value_annotation: {use_value_annotation}, label_type_number: {label_subscript_type}, direction: {direction}")
        print(f"measure_type: {measure_type}, measure_label: {measure_label}, measure_direction: {measure_direction}")
        type_number = int(type_number)
        
        comp_circuitikz_type = components_latex_info[type_number][0]
        comp_label_main = components_latex_info[type_number][1]
        comp_standard_unit = components_latex_info[type_number][2]

                                              
        labl = ""
        if control_label == -1: control_label = ""
        control_label = f"_{control_label}" if control_label != "" else ""
        if use_value_annotation:                            
            if type_number < NUM_NORMAL:
                real_value = value
                unit_mode = value_unit
                unit_scale = unit_scales[unit_mode]
                if int(note[1:]) <= 9:
                    raise NotImplementedError
                elif int(note[1:]) > 9:
                    labl = f"{int(real_value)} \\mathrm{{ {unit_scale}{comp_standard_unit} }}"
            else:
                if type_number == TYPE_VCCS or type_number == TYPE_VCVS:
                    labl = f"{value} V_{{ {control_label} }}"
                elif type_number == TYPE_CCCS or type_number == TYPE_CCVS:
                    labl = f"{value} I_{{ {control_label} }}"

        else:                           
            if type_number < NUM_NORMAL:
                if label_subscript_type == LABEL_TYPE_NUMBER:
                    if type_number == TYPE_RESISTOR:
                        labl = f"{comp_label_main}_{{ {int(label_subscript)} }}"             
                    elif type_number == TYPE_CAPACITOR:
                        labl = f"{comp_label_main}_{{ {int(label_subscript)} }}"             
                    elif type_number == TYPE_INDUCTOR:
                        labl = f"{comp_label_main}_{{ {int(label_subscript)} }}"             
                    elif type_number == TYPE_VOLTAGE_SOURCE:
                        labl = f"{comp_label_main}_{{ {int(label_subscript)} }}"             
                    elif type_number == TYPE_CURRENT_SOURCE:
                        labl = f"{comp_label_main}_{{ {int(label_subscript)} }}"             

                elif label_subscript_type == LABEL_TYPE_STRING:
                    labl = f"{comp_label_main}_{{ {label_subscript} }}"                
            
            else:
                if type_number == TYPE_VCCS or type_number == TYPE_VCVS:
                    labl = f"x_{{ {label_subscript} }} V_{{ {control_label} }}"
                elif type_number == TYPE_CCCS or type_number == TYPE_CCVS:
                    labl = f"x_{{ {label_subscript} }} I_{{ {control_label} }}"

        print(f'labl: {labl}')

                                            
        if measure_label == -1: measure_label = ""
        measure_label = f"_{{{str(measure_label)}}}" if measure_label != "" else ""
        if measure_type == MEAS_TYPE_CURRENT:
            measure_label = f"I{measure_label}"
        elif measure_type == MEAS_TYPE_VOLTAGE:
            measure_label = f"V{measure_label}"
        
                            
            
                        
        if type_number == TYPE_SHORT:
            ret = f"\\draw ({x1:.1f},{y1:.1f}) to[short] ({x2:.1f},{y2:.1f});\n"
            
            if not meas_comp_same_direction:
                    x1, y1, x2, y2 = x2, y2, x1, y1
            
            if measure_type == MEAS_TYPE_CURRENT:
                flow_dir = flow_direction[np.random.choice(range(4))]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[short, f{flow_dir}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
            print(f"ret: {ret}")
            return ret
        
                               
        elif type_number == TYPE_VOLTAGE_SOURCE:
            if int(note[1:]) < 8:
                ret =  f"\\draw ({x1:.1f},{y1:.1f}) to[V] ({x2:.1f},{y2:.1f});\n\\draw ({x1:.1f},{y1:.1f}) to [short, v=${labl}$] ({x2:.1f},{y2:.1f});\n"
            else:
                                                                                            
                if analysis_type == "ac_analysis":
                                                                                                              
                    sq_label = f"V_{{{int(label_subscript)}}}"
                    ret =  f"\\draw ({x1:.1f},{y1:.1f}) to[vsourcesquare] ({x2:.1f},{y2:.1f});\n\\draw ({x1:.1f},{y1:.1f}) to [short, v=${sq_label}$] ({x2:.1f},{y2:.1f});\n"
                else:
                    ret =  f"\\draw ({x1:.1f},{y1:.1f}) to [short] ({x2:.1f},{y2:.1f});\n\\draw ({x1:.1f},{y1:.1f}) to[rmeter, t, v=${labl}$] ({x2:.1f},{y2:.1f});\n"

            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
            if measure_type == MEAS_TYPE_CURRENT:
                flow_dir = flow_direction[np.random.choice(range(4))]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[rmeter, f{flow_dir}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
            
            return ret

                               
        elif type_number == TYPE_CURRENT_SOURCE:
            if int(note[1:]) >= 8:
                cur_mode_choices = [CUR_MODE_1, CUR_MODE_2] * 10 + [CUR_MODE_3, CUR_MODE_4] * 0 + [CUR_MODE_5, CUR_MODE_6] * 1
                cur_mode = np.random.choice(cur_mode_choices)
                print(f"cur_mode: {cur_mode} when ploting current source")
            else:
                cur_mode == CUR_MODE_1
            
            ret = f"\\draw ({x1:.1f},{y1:.1f}) to[I] ({x2:.1f},{y2:.1f});\n"

            if cur_mode == CUR_MODE_1 or cur_mode == CUR_MODE_2:
                mid = np.array([(x1+x2)/2, (y1+y2)/2])
                vector = np.array([x2-x1, y2-y1])
                normal = np.array([-vector[1], vector[0]], dtype=np.float64)
                normal /= np.linalg.norm(normal)
                if cur_mode == CUR_MODE_1:
                    new_mid = mid + base_offset*normal
                    new_mid_node = mid + node_offset*normal

                else:
                    new_mid = mid - base_offset*normal
                    new_mid_node = mid - node_offset*normal

                norm_vector = vector / np.linalg.norm(vector)
                new_start = new_mid - arrow_offset*norm_vector
                new_end = new_mid + arrow_offset*norm_vector
                ret += f"\\draw[-latexslim] ({new_start[0]:.1f},{new_start[1]:.1f}) to ({new_end[0]:.1f},{new_end[1]:.1f});\n"
                ret += f"\\node at ({new_mid_node[0]:.1f}, {new_mid_node[1]:.1f}) {{${labl}$}};\n"
                
            elif cur_mode in [CUR_MODE_3, CUR_MODE_4, CUR_MODE_5, CUR_MODE_6]:
                flow_dir = flow_direction[cur_mode-2]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[rmeter, f{flow_dir}=${labl}$] ({x2:.1f},{y2:.1f});\n"
            
            v_plot_extra = ""
            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
                v_plot_extra = "^"
            if measure_type == MEAS_TYPE_VOLTAGE:
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[rmeter, v{v_plot_extra}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
                
            return ret

                                                 
        elif type_number in [TYPE_RESISTOR, TYPE_CAPACITOR, TYPE_INDUCTOR]:
            ret = f"\\draw ({x1:.1f},{y1:.1f}) to[{comp_circuitikz_type}, l=${labl}$, ] ({x2:.1f},{y2:.1f});\n"

            v_plot_extra = ""
            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
                v_plot_extra = "^"
            if measure_type == MEAS_TYPE_VOLTAGE:
                ret +=  f"\\draw ({x1:.1f},{y1:.1f}) to[{comp_circuitikz_type}, v{v_plot_extra}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"

            elif measure_type == MEAS_TYPE_CURRENT:
                if int(note[1:]) >= 8:
                    cur_mode_choices = [CUR_MODE_1, CUR_MODE_2] * 0 + [CUR_MODE_3, CUR_MODE_4] * 1 + [CUR_MODE_5, CUR_MODE_6] * 1
                    cur_mode = np.random.choice(cur_mode_choices)
                else:
                    cur_mode = CUR_MODE_5

                if cur_mode in [CUR_MODE_1, CUR_MODE_2]:
                                                                                                  
                    mid = np.array([(x1+x2)/2, (y1+y2)/2])
                    vector = np.array([x2-x1, y2-y1])
                    normal = np.array([-vector[1], vector[0]], dtype=np.float64)
                    normal /= np.linalg.norm(normal)
                    
                    enhanced_base_offset = base_offset * 1.2
                    enhanced_node_offset = node_offset * 1.8
                    
                    if cur_mode == CUR_MODE_1:
                        new_mid = mid + enhanced_base_offset*normal
                        new_mid_node = mid + enhanced_node_offset*normal
                    else:
                        new_mid = mid - enhanced_base_offset*normal
                        new_mid_node = mid - enhanced_node_offset*normal

                    norm_vector = vector / np.linalg.norm(vector)
                    new_start = new_mid - arrow_offset*norm_vector
                    new_end = new_mid + arrow_offset*norm_vector
                    ret += f"\\draw[-latexslim] ({new_start[0]:.1f},{new_start[1]:.1f}) to ({new_end[0]:.1f},{new_end[1]:.1f});\n"
                    ret += f"\\node at ({new_mid_node[0]:.1f},{new_mid_node[1]:.1f}) {{${measure_label}$}};\n"
                
                elif cur_mode in [CUR_MODE_3, CUR_MODE_4, CUR_MODE_5, CUR_MODE_6]:
                    flow_dir = flow_direction[cur_mode-2]
                    ret += f"\\draw ({x1:.1f},{y1:.1f}) to[{comp_circuitikz_type}, f{flow_dir}=${measure_label}$] ({x2:.1f},{y2:.1f});\n" 

            return ret

                         
        elif type_number == TYPE_OPEN:
            ret = ""

            v_plot_extra = ""
            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
                v_plot_extra = "^"
            if measure_type == MEAS_TYPE_VOLTAGE:
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[open, v{v_plot_extra}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
            return ret
        
                                      
        elif type_number in [TYPE_VCVS, TYPE_CCVS]:
                                                                                        
            if not use_value_annotation:
                if type_number == TYPE_VCVS:
                    labl = f"x_{{ {label_subscript} }} V_{{ {control_label} }}"
                else:             
                    labl = f"x_{{ {label_subscript} }} I_{{ {control_label} }}"
            ret = f"\\draw ({x1:.1f},{y1:.1f}) to [short, v=${labl}$] ({x2:.1f},{y2:.1f});\n\\draw ({x1:.1f},{y1:.1f}) to[cvsource] ({x2:.1f},{y2:.1f});"

            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
            if measure_type == MEAS_TYPE_CURRENT:
                flow_dir = flow_direction[np.random.choice(range(4))]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[short, f{flow_dir}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
            
            return ret

                                                                        
        elif type_number in [TYPE_VCCS, TYPE_CCCS]:
                                                                                        
            if not use_value_annotation:
                if type_number == TYPE_VCCS:
                    labl = f"y_{{ {label_subscript} }} V_{{ {control_label} }}"
                else:             
                    labl = f"y_{{ {label_subscript} }} I_{{ {control_label} }}"
            ret = f"\\draw ({x1:.1f},{y1:.1f}) to[cisource] ({x2:.1f},{y2:.1f});\n"

            cur_mode_choices = [CUR_MODE_1, CUR_MODE_2] * 10 + [CUR_MODE_3, CUR_MODE_4] * 0 + [CUR_MODE_5, CUR_MODE_6] * 1
            cur_mode = np.random.choice(cur_mode_choices)
            print(f"cur_mode: {cur_mode} when ploting controlled current source")

            if cur_mode == CUR_MODE_1 or cur_mode == CUR_MODE_2:
                mid = np.array([(x1+x2)/2, (y1+y2)/2])
                vector = np.array([x2-x1, y2-y1])
                normal = np.array([-vector[1], vector[0]], dtype=np.float64)
                normal /= np.linalg.norm(normal)
                
                                                                               
                enhanced_base_offset = base_offset * controlled_source_offset_multiplier * 1.4                        
                enhanced_node_offset = node_offset * controlled_source_offset_multiplier * 1.6                        
                
                                                                                  
                if measure_type != MEAS_TYPE_NONE:
                    if is_horizontal:
                                                                                                            
                        if measure_type == MEAS_TYPE_CURRENT:
                            cur_mode = CUR_MODE_1                           
                            enhanced_node_offset *= 1.8                                          
                        else:
                            cur_mode = CUR_MODE_1                                           
                    elif is_vertical:
                                                                                                     
                        if measure_type == MEAS_TYPE_CURRENT:
                            cur_mode = CUR_MODE_2                          
                            enhanced_node_offset *= 1.8                                          
                        else:
                            cur_mode = CUR_MODE_2                                          
                    else:
                                                                                        
                        cur_mode = CUR_MODE_1 if (x2-x1) * (y2-y1) > 0 else CUR_MODE_2
                        enhanced_node_offset *= 1.5
                
                                                           
                if cur_mode == CUR_MODE_1:
                    new_mid = mid + enhanced_base_offset*normal
                    new_mid_node = mid + enhanced_node_offset*normal
                else:
                    new_mid = mid - enhanced_base_offset*normal
                    new_mid_node = mid - enhanced_node_offset*normal

                norm_vector = vector / np.linalg.norm(vector)
                new_start = new_mid - arrow_offset*norm_vector
                new_end = new_mid + arrow_offset*norm_vector
                ret += f"\\draw[-latexslim] ({new_start[0]:.1f},{new_start[1]:.1f}) to ({new_end[0]:.1f},{new_end[1]:.1f});\n"
                ret += f"\\node at ({new_mid_node[0]:.1f}, {new_mid_node[1]:.1f}) {{${labl}$}};\n"
                
            elif cur_mode in [CUR_MODE_3, CUR_MODE_4, CUR_MODE_5, CUR_MODE_6]:
                flow_dir = flow_direction[cur_mode-2]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[cisource, f{flow_dir}=${labl}$] ({x2:.1f},{y2:.1f});\n"

                                                                                
            v_plot_extra = ""
            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
                v_plot_extra = "^"
            if measure_type == MEAS_TYPE_VOLTAGE:
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[open, v{v_plot_extra}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
            elif measure_type == MEAS_TYPE_CURRENT:
                                                                                               
                if cur_mode == CUR_MODE_1:
                                                                                    
                    enhanced_meas_mode = CUR_MODE_6                    
                elif cur_mode == CUR_MODE_2:
                                                                                      
                    enhanced_meas_mode = CUR_MODE_5                   
                else:
                                                                   
                    enhanced_meas_mode = CUR_MODE_5 if cur_mode in [CUR_MODE_3, CUR_MODE_5] else CUR_MODE_6
                
                flow_dir = flow_direction[enhanced_meas_mode-2]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[cisource, f{flow_dir}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
                
            return ret

                                                                                      
        elif type_number == TYPE_OPAMP_INTEGRATOR:
            
                                                                        
            is_horizontal = abs(y2 - y1) < 0.1                                           
            is_vertical = abs(x2 - x1) < 0.1                                            
            
                                             
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            if is_horizontal:
                                                                                
                if x2 > x1:                                             
                    orientation = "horizontal_lr"
                else:                                             
                    orientation = "horizontal_rl"
                    
            elif is_vertical:
                                                                              
                if y2 > y1:                                             
                    orientation = "vertical_bt"
                else:                                             
                    orientation = "vertical_tb"
                    
            else:
                                                                           
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                
                if dx > dy:                                 
                    orientation = "horizontal_lr" if x2 > x1 else "horizontal_rl"
                else:                                 
                    orientation = "vertical_bt" if y2 > y1 else "vertical_tb"
            
                                                                        
                                                                               
            time_constant = int(value) if use_value_annotation else 1
            resistor_value = max(1, time_constant // 10) if time_constant > 10 else 1
            capacitor_value = max(1, time_constant - resistor_value) if time_constant > resistor_value else 1
            
                                                                      
            template_code, input_point, output_point, ground_point = draw_integrator_template(
                center_x, center_y, 
                resistor_value, capacitor_value,
                use_value_annotation, label_subscript,
                orientation
            )
            
                                                                        
            ret = template_code
            ret += f"\\draw ({x1:.1f},{y1:.1f}) to[short] ({input_point[0]:.1f},{input_point[1]:.1f});\n"
            ret += f"\\draw ({output_point[0]:.1f},{output_point[1]:.1f}) to[short] ({x2:.1f},{y2:.1f});\n"
            
                                           
            v_plot_extra = ""
            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
                v_plot_extra = "^"
            if measure_type == MEAS_TYPE_VOLTAGE:
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[open, v{v_plot_extra}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
            elif measure_type == MEAS_TYPE_CURRENT:
                flow_dir = flow_direction[np.random.choice(range(4))]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[short, f{flow_dir}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
                
            return ret
            
                                                                           
        elif type_number in [TYPE_OPAMP_INVERTING, TYPE_OPAMP_NONINVERTING, TYPE_OPAMP_BUFFER, 
                           TYPE_OPAMP_DIFFERENTIATOR, TYPE_OPAMP_SUMMING]:
            
                                                                         
            print(f"Warning: Non-integrator op-amp type {type_number} found, converting to integrator")
            return get_latex_line_draw(x1, y1, x2, y2, TYPE_OPAMP_INTEGRATOR, label_subscript, value, value_unit,
                                     use_value_annotation, style, measure_type, measure_label, measure_direction,
                                     control_label, label_subscript_type, direction, note, analysis_type)
        
                                 
        elif type_number == TYPE_BJT_SMALL_SIGNAL:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
                                                           
            if is_horizontal:
                orientation = "horizontal_lr" if x2 > x1 else "horizontal_rl"
            elif is_vertical:
                orientation = "vertical_bt" if y2 > y1 else "vertical_tb"
            else:
                                                         
                orientation = "horizontal_lr" if abs(x2-x1) > abs(y2-y1) else "vertical_bt"
            
                                                       
            if use_value_annotation:
                gm_value = float(value) if value > 0 else 50.0      
                rpi_value = 30 + np.random.randint(1, 30)     
                ro_value = 50 + np.random.randint(0, 30)     
            else:
                gm_value = 50.0                                    
                rpi_value = 20
                ro_value = 30
            
                                                                                 
            template_code, input_point, output_point = draw_bjt_small_signal_2terminal(
                center_x, center_y, gm_value, rpi_value, ro_value,
                use_value_annotation, label_subscript, orientation
            )
            
                                                                                    
            ret = template_code
            ret += f"\\draw ({x1:.1f},{y1:.1f}) to[short] ({input_point[0]:.1f},{input_point[1]:.1f});\n"
            ret += f"\\draw ({output_point[0]:.1f},{output_point[1]:.1f}) to[short] ({x2:.1f},{y2:.1f});\n"
            
                                           
            v_plot_extra = ""
            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
                v_plot_extra = "^"
            if measure_type == MEAS_TYPE_VOLTAGE:
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[open, v{v_plot_extra}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
            elif measure_type == MEAS_TYPE_CURRENT:
                flow_dir = flow_direction[np.random.choice(range(4))]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[short, f{flow_dir}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
                
            return ret
        
                                   
        elif type_number == TYPE_MOSFET_SMALL_SIGNAL:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
                                                           
            if is_horizontal:
                orientation = "horizontal_lr" if x2 > x1 else "horizontal_rl"
            elif is_vertical:
                orientation = "vertical_bt" if y2 > y1 else "vertical_tb"
            else:
                                                         
                orientation = "horizontal_lr" if abs(x2-x1) > abs(y2-y1) else "vertical_bt"
            
                                                          
            if use_value_annotation:
                gm_value = float(value) if value > 0 else 10.0      
                cgs_value = 1.0 + np.random.uniform(-0.3, 0.5)      
                cgd_value = 0.5 + np.random.uniform(-0.2, 0.3)      
                ro_value = 10000 + np.random.randint(-2000, 5000)     
            else:
                gm_value = 10.0                                    
                cgs_value = 1.0
                cgd_value = 0.5
                ro_value = 10000
            
                                          
            template_code, gate_point, drain_point, source_point = draw_mosfet_small_signal_template(
                center_x, center_y, gm_value, cgs_value, cgd_value, ro_value,
                use_value_annotation, label_subscript, orientation
            )
            
                                             
            ret = template_code
            ret += f"\\draw ({x1:.1f},{y1:.1f}) to[short] ({gate_point[0]:.1f},{gate_point[1]:.1f});\n"
            ret += f"\\draw ({drain_point[0]:.1f},{drain_point[1]:.1f}) to[short] ({x2:.1f},{y2:.1f});\n"
            
                                           
            v_plot_extra = ""
            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
                v_plot_extra = "^"
            if measure_type == MEAS_TYPE_VOLTAGE:
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[open, v{v_plot_extra}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
            elif measure_type == MEAS_TYPE_CURRENT:
                flow_dir = flow_direction[np.random.choice(range(4))]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[short, f{flow_dir}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
                
            return ret

    elif style == "american":
        pass

    elif style == "european":

        pass

    else:
        raise NotImplementedError
    pass



                        
SPICE_TEMPLATES = {
    "v4": ".title Active DC Circuit {id}\n{components}\n.END\n",
    "v5": ".title Active DC Circuit\n{components}\n\n{simulation}.END\n",
    "v6": ".title Active DC Circuit\n{components}\n\n{simulation}.END\n",
    "v7": ".title Active DC Circuit\n{components}\n\n{simulation}.END\n",
    "v8": ".title Active DC Circuit\n{components}\n\n{simulation}.END\n",
    "v9": ".title Active DC Circuit\n{components}\n\n{simulation}.END\n",
    "v10": ".title Active DC Circuit\n{components}\n\n{simulation}.end\n",
    "v11": ".title Active DC Circuit\n{components}\n\n{simulation}.end\n",
}
SPICE_PREFFIX = {
    TYPE_RESISTOR: "R",
    TYPE_CAPACITOR: "C",
    TYPE_INDUCTOR: "L",
    TYPE_VOLTAGE_SOURCE: "V",
    TYPE_CURRENT_SOURCE: "I",
    TYPE_VCCS: "G",
    TYPE_VCVS: "E",
    TYPE_CCCS: "F",
    TYPE_CCVS: "H",
    TYPE_OPEN: "",
    TYPE_SHORT: "",
                                                 
    TYPE_OPAMP_INVERTING: "X",
    TYPE_OPAMP_NONINVERTING: "X", 
    TYPE_OPAMP_BUFFER: "X",
    TYPE_OPAMP_INTEGRATOR: "X",
    TYPE_OPAMP_DIFFERENTIATOR: "X",
    TYPE_OPAMP_SUMMING: "X",
                                                          
    TYPE_BJT_SMALL_SIGNAL: "X",
    TYPE_MOSFET_SMALL_SIGNAL: "X",
}

def reassign_unique_labels(vcomp_type, hcomp_type, vcomp_label, hcomp_label, m, n):
\
\
\
       
                                                  
    component_counters = {
        TYPE_RESISTOR: 0,
        TYPE_CAPACITOR: 0,
        TYPE_INDUCTOR: 0,
        TYPE_VOLTAGE_SOURCE: 0,
        TYPE_CURRENT_SOURCE: 0,
        TYPE_VCCS: 0,
        TYPE_VCVS: 0,
        TYPE_CCCS: 0,
        TYPE_CCVS: 0,
        TYPE_OPAMP_INVERTING: 0,
        TYPE_OPAMP_NONINVERTING: 0,
        TYPE_OPAMP_BUFFER: 0,
        TYPE_OPAMP_INTEGRATOR: 0,
        TYPE_OPAMP_DIFFERENTIATOR: 0,
        TYPE_OPAMP_SUMMING: 0,
        TYPE_BJT_SMALL_SIGNAL: 0,
        TYPE_MOSFET_SMALL_SIGNAL: 0,
    }
    
                                             
    for ii in range(m-1):
        for jj in range(n):
            comp_type = vcomp_type[ii][jj]
            if comp_type in component_counters:
                component_counters[comp_type] += 1
                vcomp_label[ii][jj] = component_counters[comp_type]
    
                                               
    for ii in range(m):
        for jj in range(n-1):
            comp_type = hcomp_type[ii][jj]
            if comp_type in component_counters:
                component_counters[comp_type] += 1
                hcomp_label[ii][jj] = component_counters[comp_type]

class Circuit:

    def __init__(self, m = 3, n = 4, 
                vertical_dis = None, horizontal_dis = None,
                has_vedge = None, has_hedge = None,

                vcomp_type = None, hcomp_type = None,
                vcomp_label = None, hcomp_label = None,                                        
                vcomp_value = None, hcomp_value = None,
                vcomp_value_unit = None, hcomp_value_unit = None,
                vcomp_direction = None, hcomp_direction = None,

                vcomp_measure = None, hcomp_measure = None,
                vcomp_measure_label = None, hcomp_measure_label = None,                                
                vcomp_measure_direction = None, hcomp_measure_direction = None,

                vcomp_control_meas_label = None, hcomp_control_meas_label = None,                                                           

                use_value_annotation = False,
                note = "v1",
                id = "",
                label_numerical_subscript = True,
                rlc = False,
                no_meas = False,
                ):
        self.m = m
        self.n = n
        self.vertical_dis = np.arange(m)*4.0 if vertical_dis is None else vertical_dis
        self.horizontal_dis = np.arange(n)*3.0 if horizontal_dis is None else horizontal_dis

        self.has_vedge = np.ones((m-1, n)) if has_vedge is None else has_vedge         
        self.has_hedge = np.ones((m, n-1)) if has_hedge is None else has_hedge

        self.vcomp_type = np.zeros((m-1, n)) if vcomp_type is None else vcomp_type
        self.hcomp_type = np.zeros((m, n-1)) if hcomp_type is None else hcomp_type
        self.vcomp_label = np.ones((m-1, n)) if vcomp_label is None else vcomp_label
        self.hcomp_label = np.ones((m, n-1)) if hcomp_label is None else hcomp_label
        self.vcomp_value = np.zeros((m-1, n)) if vcomp_value is None else vcomp_value
        self.hcomp_value = np.zeros((m, n-1)) if hcomp_value is None else hcomp_value
        self.vcomp_value_unit = np.zeros((m-1, n)) if vcomp_value_unit is None else vcomp_value_unit
        self.hcomp_value_unit = np.zeros((m, n-1)) if hcomp_value_unit is None else hcomp_value_unit

        self.vcomp_direction = np.zeros((m-1, n)) if vcomp_direction is None else vcomp_direction                         
        self.hcomp_direction = np.zeros((m, n-1)) if hcomp_direction is None else hcomp_direction                         

        self.vcomp_measure = np.zeros((m-1, n)) if vcomp_measure is None else vcomp_measure
        self.hcomp_measure = np.zeros((m, n-1)) if hcomp_measure is None else hcomp_measure

        self.vcomp_measure_label = np.zeros((m-1, n)) if vcomp_measure_label is None else vcomp_measure_label
        self.hcomp_measure_label = np.zeros((m, n-1)) if hcomp_measure_label is None else hcomp_measure_label

        self.vcomp_measure_direction = np.zeros((m-1, n)) if vcomp_measure_direction is None else vcomp_measure_direction
        self.hcomp_measure_direction = np.zeros((m, n-1)) if hcomp_measure_direction is None else hcomp_measure_direction

        self.vcomp_control_meas_label = np.zeros((m-1, n)) if vcomp_control_meas_label is None else vcomp_control_meas_label
        self.hcomp_control_meas_label = np.zeros((m, n-1)) if hcomp_control_meas_label is None else hcomp_control_meas_label

        self.use_value_annotation = use_value_annotation                                                                                 

        self.latex_font_size = "\\large"

        if self.use_value_annotation:
            self.label_numerical_subscript = True
        else:
            self.label_numerical_subscript = label_numerical_subscript

        self.rlc = rlc
        self.no_meas = no_meas

                                                                                                      
        self.controller_voltage_labels = set()
        self.controller_current_labels = set()
        
        self.note = note
        self.id = id

        self._init_degree()                    
        self._check_circuit_valid_by_degree()                                           
        self._init_netlist()                                                                  
        
                                                                                              
        if getattr(self, 'valid', False):
            self._collect_controller_measure_labels()
        
                                                                
        if self.valid:
            self._calculate_adaptive_spacing()
            self._adjust_font_size()
            self._optimize_measurement_placement()

    def _collect_controller_measure_labels(self):
        vol_labels = set()
        cur_labels = set()
        for br in self.branches:
            try:
                ctl = int(br.get('control_measure_label', -1))
            except Exception:
                ctl = -1
            if ctl == -1:
                continue
            if br.get('type') in [TYPE_VCVS, TYPE_VCCS]:
                vol_labels.add(ctl)
            if br.get('type') in [TYPE_CCCS, TYPE_CCVS]:
                cur_labels.add(ctl)
        self.controller_voltage_labels = vol_labels
        self.controller_current_labels = cur_labels

    def _init_degree(self):
        self.degree = np.zeros((self.m, self.n))

        for i in range(self.m):
            for j in range(self.n):
                if j>0:
                    self.degree[i][j] += (self.has_hedge[i][j-1] and self.hcomp_type[i][j-1] != TYPE_OPEN)
                if j<self.n-1:
                    self.degree[i][j] += (self.has_hedge[i][j] and self.hcomp_type[i][j] != TYPE_OPEN)
                if i>0:
                    self.degree[i][j] += (self.has_vedge[i-1][j] and self.vcomp_type[i-1][j] != TYPE_OPEN)
                if i<self.m-1:
                    self.degree[i][j] += (self.has_vedge[i][j] and self.vcomp_type[i][j] != TYPE_OPEN)

        self._degree_init = True
    
    def _get_grid_nodes(self):
        m, n = self.m, self.n
        visited = [[False]*n for _ in range(m)]                       
        components = []                                  
        
        def dfs(i, j, component):
            if i < 0 or i >= m or j < 0 or j >= n or visited[i][j]:
                return
            visited[i][j] = True
            component.append((i, j))
            
                                        
            if i > 0 and self.has_vedge[i-1][j] and self.vcomp_type[i-1][j] == TYPE_SHORT and self.vcomp_measure[i-1][j] == MEAS_TYPE_NONE: dfs(i-1, j, component)
            if j > 0 and self.has_hedge[i][j-1] and self.hcomp_type[i][j-1] == TYPE_SHORT and self.hcomp_measure[i][j-1] == MEAS_TYPE_NONE: dfs(i, j-1, component)
            if i < m-1 and self.has_vedge[i][j] and self.vcomp_type[i][j] == TYPE_SHORT and self.vcomp_measure[i][j] == MEAS_TYPE_NONE: dfs(i+1, j, component)
            if j < n-1 and self.has_hedge[i][j] and self.hcomp_type[i][j] == TYPE_SHORT and self.hcomp_measure[i][j] == MEAS_TYPE_NONE: dfs(i, j+1, component)
            
        for i in range(m):
            for j in range(n):
                if not visited[i][j]: 
                    component = []
                    dfs(i, j, component)
                    components.append(component)

        print(f"components: {components}")
                                    
        self.nodes = [f"{i}" for i in range(len(components))]
        grid_nodes = np.zeros((m, n))               
        for i in range(len(components)):
            for x, y in components[i]:
                grid_nodes[x][y] = i
        
        return grid_nodes
    
    def _check_conflict_component_measure(self, comp_type, comp_measure):
        conflict_pairs = [
            (TYPE_SHORT, MEAS_TYPE_VOLTAGE),
            (TYPE_OPEN, MEAS_TYPE_CURRENT), 
            (TYPE_VOLTAGE_SOURCE, MEAS_TYPE_VOLTAGE),
            (TYPE_VCVS, MEAS_TYPE_VOLTAGE),
            (TYPE_CCVS, MEAS_TYPE_VOLTAGE),
            (TYPE_CURRENT_SOURCE, MEAS_TYPE_CURRENT),
            (TYPE_VCCS, MEAS_TYPE_CURRENT),
            (TYPE_CCCS, MEAS_TYPE_CURRENT),
        ]
        for pair in conflict_pairs:
            if comp_type == pair[0] and comp_measure == pair[1]:
                return True
        return False

    def init_netlist(self):
        return self._init_netlist()

    def _init_netlist(self):
\
\
\
           
        assert self._degree_init, "degree not initialized"

        self.nodes = []
        self.branches = []
        
        self.grid_nodes = self._get_grid_nodes()
        print(f"Grid Nodes:\n{self.grid_nodes}\n\n")

        print("self.hcomp_type: \n", self.hcomp_type)
        print("self.has_hedge: \n", self.has_hedge)

        add_order = 0
        for i in range(self.m):
            for j in range(self.n):
                if int(self.note[1:]) <= 9:
                    raise NotImplementedError
                elif int(self.note[1:]) > 9:
                    print(f"({i}, {j}) / ({self.m}, {self.n})")
                    if j < self.n-1 and self.has_hedge[i][j]:
                        print(f"({i}, {j}) has hedge")
                        assert self.hcomp_type[i][j] != TYPE_OPEN, f"open circuit should not be in the netlist, {self.hcomp_type[i][j]}"
                        if self.grid_nodes[i][j] == self.grid_nodes[i][j+1]:
                            if self.hcomp_type[i][j] != TYPE_SHORT:
                                print("invalid circuit, some components are shorted")
                                self.valid = False
                                return False
                        
                        else:
                            n1 = f"{int(self.grid_nodes[i][j])}"
                            n2 = f"{int(self.grid_nodes[i][j+1])}"
                            if self.hcomp_direction[i][j]:
                                n1, n2 = n2, n1
                            new_branch = {
                                "n1": n1,
                                "n2": n2,
                                "type": self.hcomp_type[i][j],
                                "label": self.hcomp_label[i][j],
                                "value": self.hcomp_value[i][j],
                                "value_unit": self.hcomp_value_unit[i][j],
                                "measure": self.hcomp_measure[i][j],
                                "measure_label": self.hcomp_measure_label[i][j],
                                "meas_comp_same_direction": self.hcomp_measure_direction[i][j] == self.hcomp_direction[i][j],
                                "control_measure_label": self.hcomp_control_meas_label[i][j],
                                "info": "",
                                "order": add_order
                            }

                            if i == 1 and j == 1:
                                print(f"new_branch: {new_branch} on [1, 1]")

                            if self._check_conflict_component_measure(self.hcomp_type[i][j], self.hcomp_measure[i][j]):
                                print("invalid circuit, conflict between component type and measure type")
                                self.valid = False
                                return False
                        
                            self.branches.append(new_branch)
                            add_order += 1
                    
                    if i < self.m-1 and self.has_vedge[i][j]:
                        print(f"({i}, {j}) has vedge")
                        if self.grid_nodes[i][j] == self.grid_nodes[i+1][j]:
                            if self.vcomp_type[i][j] != TYPE_SHORT:
                                print("invalid circuit, some components are shorted")
                                self.valid = False
                                return False
                            
                        else:            
                            n1 = f"{int(self.grid_nodes[i][j])}"
                            n2 = f"{int(self.grid_nodes[i+1][j])}"
                            if self.vcomp_direction[i][j]:
                                n1, n2 = n2, n1
                            new_branch = {
                                "n1": n1,
                                "n2": n2,
                                "type": self.vcomp_type[i][j],
                                "label": self.vcomp_label[i][j],
                                "value": self.vcomp_value[i][j],
                                "value_unit": self.vcomp_value_unit[i][j],
                                "measure": self.vcomp_measure[i][j],
                                "measure_label": self.vcomp_measure_label[i][j],
                                "meas_comp_same_direction": self.vcomp_measure_direction[i][j] == self.vcomp_direction[i][j],
                                "control_measure_label": self.vcomp_control_meas_label[i][j],
                                "info": "",
                                "order": add_order
                            }

                            if self._check_conflict_component_measure(self.vcomp_type[i][j], self.vcomp_measure[i][j]):
                                print("invalid circuit, conflict between component type and measure type")
                                self.valid = False
                                return False

                            self.branches.append(new_branch)
                            add_order += 1

        for br in self.branches:
            tmp = [(b['n1'], b['n2']) for b in self.branches if b['measure_label'] == br['control_measure_label'] and b['measure'] == MEAS_TYPE_VOLTAGE]
            if len(tmp) != 1:
                print(f"Controlled Source should have one and only one voltage measurement, but got {len(tmp)}, {br['control_measure_label']}")
                self.valid = False
                return False
        
                                                                                                   
                            
        print(f"init netlist done, get branches: {self.branches}")

                                                                            
        zero_order = True
        for br in self.branches:
            if br["type"] in [TYPE_CAPACITOR, TYPE_INDUCTOR]:
                zero_order = False
                break
        self.analysis_type = "dc_analysis" if zero_order else "ac_analysis"
        print(f"[init_netlist] Circuit analysis type: {self.analysis_type}")

                                                                            
                                
                                                       
                                                                         
                                                                             
                                                                             
                                               
                                                                            
                                                                   
        vs_neg_node_old = None
        for b in self.branches:
            if b["type"] == TYPE_VOLTAGE_SOURCE:
                vs_neg_node_old = str(b["n2"])
                break
        if vs_neg_node_old is not None:
                            
            all_nodes = set()
            for b in self.branches:
                all_nodes.add(str(b["n1"]))
                all_nodes.add(str(b["n2"]))

                                                                      
                                                 
            node_map = {}
            for node in all_nodes:
                if node == vs_neg_node_old:
                    node_map[node] = '0'
                else:
                    node_map[node] = str(int(node) + 1)

                                       
            for b in self.branches:
                b["n1"] = node_map[str(b["n1"])]
                b["n2"] = node_map[str(b["n2"])]

                                                                            
            old_neg_idx = int(vs_neg_node_old)
                             
            self.grid_nodes = self.grid_nodes.astype(int) + 1
                                                                                
            self.grid_nodes[self.grid_nodes == (old_neg_idx + 1)] = 0

                                                                         
            unique_node_ids = sorted(list(set(int(x) for x in self.grid_nodes.flatten().tolist())))
            self.nodes = [str(i) for i in unique_node_ids]

        return True
        pass
    
    def _to_SPICE(self):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        spice_str = ""
        
                                                                   
        zero_order = True
        for br in self.branches:
            if br["type"] in [TYPE_CAPACITOR, TYPE_INDUCTOR]:
                zero_order = False
                break
        
                                                        
        self.analysis_type = "dc_analysis" if zero_order else "ac_analysis"
                                                                   
        if getattr(self, 'rlc', False):
            self.analysis_type = "ac_analysis"
        print(f"Circuit analysis type: {self.analysis_type}")

                            
        if int(self.note[1:]) <= 9:
            raise NotImplementedError
        elif int(self.note[1:]) > 9:
            sorted_branches = sorted(self.branches, key=lambda x: x["order"])
            vmeas_counter = 0                                                      
            
                                                                                                  
                                                          
            ground_original_node = None
            def _map_node_name(node):
                return str(node)
            
                                                                            
            comp_counters = {
                TYPE_RESISTOR: 0,
                TYPE_CAPACITOR: 0, 
                TYPE_INDUCTOR: 0,
                TYPE_VOLTAGE_SOURCE: 0,
                TYPE_CURRENT_SOURCE: 0,
                TYPE_VCCS: 0,
                TYPE_VCVS: 0,
                TYPE_CCCS: 0,
                TYPE_CCVS: 0
            }
            
                                                                            
            used_device_names: set[str] = set()
            
            for br in sorted_branches:
                meas_comp_same_direction = br["meas_comp_same_direction"]
                ms_label_str = "" if br["measure_label"] == -1 else str(int(br["measure_label"]))
                ctr_ms_label_str = "" if br["control_measure_label"] == -1 else str(int(br["control_measure_label"])) 

                value_write = str(int(br["value"]))+unit_scales[br["value_unit"]] if self.use_value_annotation else "<Empty>"
                
                                                     
                type_str = SPICE_PREFFIX[br['type']]
                                                                          
                                                                          
                                                                           
                                                                           
                                 
                label_num = int(br["label"]) if br["label"] != -1 else 1
                device_name = f"{type_str}{label_num}"

                                                                      
                                                                            
                                                                       
                while device_name in used_device_names:
                    comp_counters[br['type']] += 1
                    device_name = f"{type_str}{label_num}_{comp_counters[br['type']]}"

                used_device_names.add(device_name)

                print(br["type"], br["label"], br["n1"], br["n2"], br["value"], br["value_unit"])
                print(f"Device name: {device_name}")

                if br["type"] == TYPE_SHORT:
                    assert br["measure"] == MEAS_TYPE_CURRENT, f"short circuit should be measured by current, {br}"
                    vmeas_counter += 1
                    vmeas_str = f"VI{vmeas_counter}"
                    spice_str += "%s %s %s %s\n" % (vmeas_str, _map_node_name(br["n1"]), _map_node_name(br["n2"]), 0)
                
                if br["type"] in [TYPE_VOLTAGE_SOURCE, TYPE_CURRENT_SOURCE, TYPE_RESISTOR, TYPE_CAPACITOR, TYPE_INDUCTOR]:
                    if br["measure"] == MEAS_TYPE_CURRENT:
                        mid_node = "N%s%s" % (_map_node_name(br['n1']), _map_node_name(br['n2']))
                        vmeas_counter += 1
                        vmeas_str = f"VI{vmeas_counter}"
                        
                                                                                 
                        if br["type"] == TYPE_VOLTAGE_SOURCE:
                            if self.analysis_type == "ac_analysis":
                                                                                      
                                amplitude = device_name if not self.use_value_annotation else str(int(br["value"]))
                                spice_str += "%s %s %s step %s\n" % (device_name, _map_node_name(br["n1"]), mid_node, amplitude)
                            else:
                                                              
                                spice_str += "%s %s %s %s\n" % (device_name, _map_node_name(br["n1"]), mid_node, value_write)
                        else:
                            spice_str += "%s %s %s %s\n" %  (device_name,  _map_node_name(br["n1"]),   mid_node,   value_write)
                        
                        spice_str += "%s %s %s 0\n" %       (vmeas_str,             mid_node,   _map_node_name(br["n2"])) if meas_comp_same_direction \
                                else "%s %s %s 0\n" % (vmeas_str, _map_node_name(br["n2"]), mid_node)
                    else:
                                                                                 
                        if br["type"] == TYPE_VOLTAGE_SOURCE:
                            if self.analysis_type == "ac_analysis":
                                                                                      
                                amplitude = device_name if not self.use_value_annotation else str(int(br["value"]))
                                spice_str += "%s %s %s step %s\n" % (device_name, _map_node_name(br["n1"]), _map_node_name(br["n2"]), amplitude)
                            else:
                                                              
                                spice_str += "%s %s %s %s\n" % (device_name, _map_node_name(br["n1"]), _map_node_name(br["n2"]), value_write)
                        else:
                            spice_str += "%s %s %s %s\n" %   (device_name,  _map_node_name(br["n1"]),   _map_node_name(br["n2"]),   value_write)

                if br["type"] in [TYPE_CCVS, TYPE_CCCS]:                 

                    tmp = [b for b in self.branches if b['measure_label'] == br['control_measure_label'] and b['measure'] == MEAS_TYPE_CURRENT]
                    assert len(tmp) == 1, "Controlled Source should have one and only one current measurement, but got %d, %d" % (len(tmp), br['control_measure_label'])

                                                                                         
                    control_branch = tmp[0]
                    control_vmeas_counter = 0
                    for temp_br in sorted_branches:
                        if temp_br["measure"] == MEAS_TYPE_CURRENT:
                            control_vmeas_counter += 1
                            if temp_br == control_branch:
                                break
                    control_measure_str = f"VI{control_vmeas_counter}"

                                                                            
                    gain_value = value_write
                    if not self.use_value_annotation:
                        gain_value = f"y_{int(br['label'])}" if br["type"] == TYPE_CCCS else f"x_{int(br['label'])}"

                    if br["measure"] == MEAS_TYPE_CURRENT:
                        mid_node = "N%s%s" % (_map_node_name(br['n1']), _map_node_name(br['n2']))
                        vmeas_counter += 1
                        vmeas_str = f"VI{vmeas_counter}"
                        spice_str += "%s %s %s %s %s\n" %  (device_name,  _map_node_name(br["n1"]),   mid_node,   control_measure_str,  gain_value)
                        spice_str += "%s %s %s 0\n" %       (vmeas_str,             mid_node,   _map_node_name(br["n2"])) if meas_comp_same_direction \
                                else "%s %s %s 0\n" % (vmeas_str, _map_node_name(br["n2"]), mid_node)
                    else:
                        spice_str += "%s %s %s %s %s\n" %   (device_name,  _map_node_name(br["n1"]),   _map_node_name(br["n2"]),  control_measure_str,   gain_value)
            
                if br["type"] in [TYPE_VCVS, TYPE_VCCS]:    

                    tmp = [(b['n1'], b['n2']) for b in self.branches if b['measure_label'] == br['control_measure_label'] and b['measure'] == MEAS_TYPE_VOLTAGE]
                    assert len(tmp) == 1, "Controlled Source should have one and only one voltage measurement, but got %d, %d" % (len(tmp), br['control_measure_label'])

                    control_n1, control_n2 = tmp[0]

                    gain_value = value_write
                    if not self.use_value_annotation:
                        gain_value = f"y_{int(br['label'])}" if br["type"] == TYPE_VCCS else f"x_{int(br['label'])}"

                    if br["measure"] == MEAS_TYPE_CURRENT:
                        mid_node = "N%s%s" % (_map_node_name(br['n1']), _map_node_name(br['n2']))
                        vmeas_counter += 1
                        vmeas_str = f"VI{vmeas_counter}"
                        spice_str += "%s %s %s %s %s %s\n" %  (device_name,  _map_node_name(br["n1"]),   mid_node,   _map_node_name(control_n1),  _map_node_name(control_n2),  gain_value)
                        spice_str += "%s %s %s 0\n" %       (vmeas_str,             mid_node,   _map_node_name(br["n2"])) if meas_comp_same_direction \
                                else "%s %s %s 0\n" % (vmeas_str, _map_node_name(br["n2"]), mid_node)
                    else:
                        spice_str += "%s %s %s %s %s %s\n" %  (device_name,  _map_node_name(br["n1"]),   _map_node_name(br["n2"]),   _map_node_name(control_n1),  _map_node_name(control_n2),  gain_value)

                                                                                     
                if br["type"] == TYPE_OPAMP_INTEGRATOR:
                    
                                                                                 
                    
                                                                                        
                    time_constant = int(br["value"]) if self.use_value_annotation else 1
                    resistor_value = max(1, time_constant // 10) if time_constant > 10 else 1
                    capacitor_value = max(1, time_constant - resistor_value) if time_constant > resistor_value else 1
                    
                                                                                        
                    opamp_num = device_name[1:]                                   
                    unique_r_name = f"Rint{opamp_num}"                     
                    unique_c_name = f"Cint{opamp_num}"                       
                    unique_e_name = f"Eint{opamp_num}"                     
                    
                                                                       
                    input_node = br["n1"]                          
                    output_node = br["n2"]                          
                                                                                               
                    inverting_node = str(30 + int(opamp_num))                           
                    
                    spice_str += f"* Integrator template: R-C feedback with op-amp\n"
                    
                                                                       
                    if self.use_value_annotation:
                                          
                        r_value_str = str(resistor_value)
                        c_value_str = f"{capacitor_value}e-6"                                                
                    else:
                                                                                        
                        r_value_str = "<Empty>"
                        c_value_str = "<Empty>"
                    
                                                                               
                    spice_str += f"{unique_r_name} {input_node} {inverting_node} {r_value_str}\n"
                    
                                                                                
                    spice_str += f"{unique_c_name} {output_node} {inverting_node} {c_value_str}\n"
                    
                                                                          
                                              
                                                                                  
                    if self.use_value_annotation:
                        differential_gain = 100000                            
                    else:
                        differential_gain = "Ad"                                
                    spice_str += f"{unique_e_name} {output_node} 0 0 {inverting_node} {differential_gain}\n"
                    
                                                       
                    if br["measure"] == MEAS_TYPE_CURRENT:
                        mid_node = f"Nmeas{opamp_num}"
                        vmeas_counter += 1
                        vmeas_str = f"VI{vmeas_counter}"
                                                                                 
                        spice_str += f"{vmeas_str} {mid_node} {output_node} 0\n"
                                                                                               
                        spice_str = spice_str.replace(f"{unique_e_name} {output_node} 0 0", f"{unique_e_name} {mid_node} 0 0")
                
                                                                                     
                elif br["type"] in [TYPE_OPAMP_INVERTING, TYPE_OPAMP_NONINVERTING, TYPE_OPAMP_BUFFER, 
                                  TYPE_OPAMP_DIFFERENTIATOR, TYPE_OPAMP_SUMMING]:
                    
                                                                                                  
                    print(f"Warning: Non-integrator op-amp type {br['type']} found in SPICE generation")
                    if self.use_value_annotation:
                        gain = int(br["value"])
                    else:
                        gain = "Ad"                                      
                    spice_str += f"* Fallback: Simple op-amp (should be integrator)\n"
                    fallback_name = f"E{device_name[1:]}"
                    spice_str += f"{fallback_name} {br['n2']} 0 0 {br['n1']} {gain}\n"
                    
                                                       
                    if br["measure"] == MEAS_TYPE_CURRENT:
                        mid_node = f"N{br['n1']}{br['n2']}"
                        vmeas_counter += 1
                        vmeas_str = f"VI{vmeas_counter}"
                        spice_str += f"{vmeas_str} {mid_node} {br['n2']} 0\n"
                        spice_str = spice_str.replace(f"{fallback_name} {br['n2']} 0 0", f"{fallback_name} {mid_node} 0 0")

                            
        if int(self.note[1:]) <= 9:
            raise NotImplementedError
        elif int(self.note[1:]) > 9:
                                                                      
            if self.analysis_type == "dc_analysis":                            
                sim_str = ".control\nop\n"
                current_meas_counter = 0                                                  
                
                for br in self.branches:
                    if br["measure_label"] == -1:
                        ms_label_str = ""
                    else:
                        ms_label_str = str(int(br["measure_label"]))

                    if br["measure"] == MEAS_TYPE_VOLTAGE:
                        print(f"#n1: {br['n1']}, n2: {br['n2']}")
                                                                                                                     
                        meas_n1, meas_n2 = br["n1"], br["n2"]
                        if not br["meas_comp_same_direction"]:
                            meas_n1, meas_n2 = meas_n2, meas_n1
                        if str(meas_n1) == '0':
                            sim_str += "print -v(%s) ; measurement of U%s\n" % (meas_n2, ms_label_str)
                        elif str(meas_n2) == '0':
                            sim_str += "print v(%s) ; measurement of U%s\n" % (meas_n1, ms_label_str)
                        else:
                            sim_str += "print v(%s, %s) ; measurement of U%s\n" % (meas_n1, meas_n2, ms_label_str)
                    elif br["measure"] == MEAS_TYPE_CURRENT:
                        print('#')
                                                                                                                                                            
                        current_meas_counter += 1
                        vmeas_str = f"VI{current_meas_counter}"
                        sim_str += "print i(%s) ; measurement of I%s\n" % (vmeas_str, ms_label_str)
                sim_str += ".endc\n"
                print(f"spice_str: {spice_str}, \n\nsim_str: {sim_str}\n\n")
                        
                spice_str = SPICE_TEMPLATES[self.note].format(components=spice_str, simulation=sim_str)   

            else:                                 
                                                                            
                print("Generating AC analysis for RLC circuit")
                
                                                                  
                start_freq = "1"                     
                stop_freq = "100k"                       
                points_per_decade = "10"                        
                
                sim_str = f".control\nac dec {points_per_decade} {start_freq} {stop_freq}\n"
                current_meas_counter = 0                                                  
                
                                                                                           
                for br in self.branches:
                    if br["measure_label"] == -1:
                        ms_label_str = ""
                    else:
                        ms_label_str = str(int(br["measure_label"]))

                    if br["measure"] == MEAS_TYPE_VOLTAGE:
                        print(f"#AC voltage measurement: n1: {br['n1']}, n2: {br['n2']}")
                        meas_n1, meas_n2 = br["n1"], br["n2"]
                        if not br["meas_comp_same_direction"]:
                            meas_n1, meas_n2 = meas_n2, meas_n1
                        if str(meas_n1) == '0':
                                                                      
                            sim_str += "print vm(%s) vp(%s) ; AC magnitude and phase of U%s\n" % (meas_n2, meas_n2, ms_label_str)
                        elif str(meas_n2) == '0':
                            sim_str += "print vm(%s) vp(%s) ; AC magnitude and phase of U%s\n" % (meas_n1, meas_n1, ms_label_str)
                        else:
                                                                             
                            sim_str += "print vm(%s,%s) vp(%s,%s) ; AC magnitude and phase of U%s\n" % (meas_n1, meas_n2, meas_n1, meas_n2, ms_label_str)
                    elif br["measure"] == MEAS_TYPE_CURRENT:
                        print('#AC current measurement')
                        current_meas_counter += 1
                        vmeas_str = f"VI{current_meas_counter}"
                                                                          
                        sim_str += "print im(%s) ip(%s) ; AC magnitude and phase of I%s\n" % (vmeas_str, vmeas_str, ms_label_str)
                
                sim_str += ".endc\n"
                print(f"AC analysis: freq range {start_freq}Hz to {stop_freq}Hz, {points_per_decade} points/decade")
                print(f"spice_str: {spice_str}, \n\nsim_str: {sim_str}\n\n")
                spice_str = SPICE_TEMPLATES[self.note].format(components=spice_str, simulation=sim_str)        
        else:
            raise NotImplementedError

        return spice_str
        pass

    def _check_circuit_valid_by_degree(self):

        assert self._degree_init, "degree not initialized"

                                                           
        self.valid = True
        for i in range(self.m):
            for j in range(self.n):
                if self.degree[i][j] == 1:
                    print("invalid cricuit")
                    self.valid = False
        
                                                                                         
        if self.valid:
            print("valid circuit")
        else:
            print("invalid circuit")

    def _draw_vertical_edge(self, i, j):
        if ((i>=0 and i<self.m-1) and (j>=0 and j<self.n)) and self.has_vedge[i][j]:
            if int(self.note[1:]) < 9:               
                raise NotImplementedError
            else:
                                                                               
                v_meas_type = self.vcomp_measure[i][j]
                v_meas_label = self.vcomp_measure_label[i][j]
                show_meas = (
                    (v_meas_type == MEAS_TYPE_VOLTAGE and int(v_meas_label) in self.controller_voltage_labels) or
                    (v_meas_type == MEAS_TYPE_CURRENT and int(v_meas_label) in self.controller_current_labels)
                )
                new_line = get_latex_line_draw(self.horizontal_dis[j], self.vertical_dis[i], self.horizontal_dis[j], self.vertical_dis[i+1],
                                                self.vcomp_type[i][j], 
                                                self.vcomp_label[i][j], 
                                                self.vcomp_value[i][j], 
                                                self.vcomp_value_unit[i][j],
                                                self.use_value_annotation,
                                                measure_type=(v_meas_type if show_meas else MEAS_TYPE_NONE), 
                                                measure_label=(v_meas_label if show_meas else -1),
                                                measure_direction=self.vcomp_measure_direction[i][j],
                                                direction=self.vcomp_direction[i][j],
                                                label_subscript_type=int(not self.label_numerical_subscript),
                                                control_label=self.vcomp_control_meas_label[i][j],
                                                note=self.note,
                                                analysis_type=getattr(self, 'analysis_type', 'dc_analysis')
                                            )
            return new_line
        else:
            return ""
        
    def _draw_horizontal_edge(self, i, j):
        if ((i>=0 and i<self.m) and (j>=0 and j<self.n-1)) and self.has_hedge[i][j]:
            if int(self.note[1:]) < 9:               
                raise NotImplementedError
            else:
                                                                               
                h_meas_type = self.hcomp_measure[i][j]
                h_meas_label = self.hcomp_measure_label[i][j]
                show_meas = (
                    (h_meas_type == MEAS_TYPE_VOLTAGE and int(h_meas_label) in self.controller_voltage_labels) or
                    (h_meas_type == MEAS_TYPE_CURRENT and int(h_meas_label) in self.controller_current_labels)
                )
                new_line = get_latex_line_draw(self.horizontal_dis[j], self.vertical_dis[i], self.horizontal_dis[j+1], self.vertical_dis[i],
                                                self.hcomp_type[i][j], 
                                                self.hcomp_label[i][j], 
                                                self.hcomp_value[i][j],
                                                self.hcomp_value_unit[i][j],
                                                self.use_value_annotation,
                                                measure_type=(h_meas_type if show_meas else MEAS_TYPE_NONE), 
                                                measure_label=(h_meas_label if show_meas else -1),
                                                measure_direction=self.hcomp_measure_direction[i][j],
                                                direction=self.hcomp_direction[i][j],
                                                label_subscript_type=int(not self.label_numerical_subscript),
                                                control_label=self.hcomp_control_meas_label[i][j],
                                                note=self.note,
                                                analysis_type=getattr(self, 'analysis_type', 'dc_analysis')
                                            )
            return new_line
        else: 
            return ""
        
    def to_latex(self):
                                                                
                                       
        if int(self.note[1:]) <= 9:
            raise NotImplementedError
        elif int(self.note[1:]) > 9:
            latex_template = LATEX_TEMPLATES["v11"]
        else:
            raise NotImplementedError
        
        latex_code_main = ""
        for i in range(self.m):
            for j in range(self.n):
                latex_code_main += self._draw_horizontal_edge(i,j)
                latex_code_main += self._draw_vertical_edge(i,j)
        
                                                              
                                                                                    
        node_label_code = ""
        for i in range(self.m):
            for j in range(self.n):
                node_num = int(self.grid_nodes[i][j])
                x_coord = self.horizontal_dis[j]
                y_coord = self.vertical_dis[i]
                
                                                                             
                is_connected = False
                
                                                                 
                if i > 0 and self.has_vedge[i-1][j]:              
                    is_connected = True
                if i < self.m-1 and self.has_vedge[i][j]:                
                    is_connected = True
                if j > 0 and self.has_hedge[i][j-1]:                
                    is_connected = True
                if j < self.n-1 and self.has_hedge[i][j]:                 
                    is_connected = True
                
                if is_connected:
                                                                                       
                    node_label_code += f"\\node[circle, draw=blue, fill=white, inner sep=2pt] at ({x_coord:.1f},{y_coord:.1f}) {{\\textcolor{{blue}}{{\\tiny {node_num}}}}};\n"
        
        latex_code_main += node_label_code
        latex_code = latex_template.replace("<main>", latex_code_main)
        
        if int(self.note[1:]) >= 8:
            latex_code = latex_code.replace("<font>", self.latex_font_size)

        return latex_code

    def _calculate_adaptive_spacing(self):
                                                                                        
        
                                                
        measurement_count = 0
        complex_component_count = 0
        integrator_count = 0                                             
        
        for i in range(self.m-1):
            for j in range(self.n):
                if self.has_vedge[i][j]:
                    if self.vcomp_measure[i][j] != MEAS_TYPE_NONE:
                        measurement_count += 1
                    if self.vcomp_type[i][j] in [TYPE_VCCS, TYPE_VCVS, TYPE_CCCS, TYPE_CCVS]:
                        complex_component_count += 1
                    if self.vcomp_type[i][j] == TYPE_OPAMP_INTEGRATOR:
                        integrator_count += 1                                             
                        
        for i in range(self.m):
            for j in range(self.n-1):
                if self.has_hedge[i][j]:
                    if self.hcomp_measure[i][j] != MEAS_TYPE_NONE:
                        measurement_count += 1
                    if self.hcomp_type[i][j] in [TYPE_VCCS, TYPE_VCVS, TYPE_CCCS, TYPE_CCVS]:
                        complex_component_count += 1
                    if self.hcomp_type[i][j] == TYPE_OPAMP_INTEGRATOR:
                        integrator_count += 1                                             
        
                                                           
        base_h_spacing = 4.5 if integrator_count > 0 else 3.5
        base_v_spacing = 4.5 if integrator_count > 0 else 3.5
        
                                                                            
        total_edges = len(self.horizontal_dis) * len(self.vertical_dis)
        complexity_factor = 1 + (measurement_count + complex_component_count + integrator_count * 2) / max(total_edges, 1)
        
                                                                                 
        integrator_multiplier = 1.3 if integrator_count > 0 else 1.0
        adaptive_h_spacing = base_h_spacing * max(1.2, complexity_factor * 0.9) * integrator_multiplier
        adaptive_v_spacing = base_v_spacing * max(1.2, complexity_factor * 0.9) * integrator_multiplier
        
                                                     
        for i in range(len(self.horizontal_dis)):
            if i > 0:
                self.horizontal_dis[i] = self.horizontal_dis[i-1] + adaptive_h_spacing + np.random.uniform(-0.4, 0.4)
        
        for i in range(len(self.vertical_dis)):
            if i > 0:
                self.vertical_dis[i] = self.vertical_dis[i-1] + adaptive_v_spacing + np.random.uniform(-0.4, 0.4)
        
        print(f"Adaptive spacing applied: h={adaptive_h_spacing:.2f}, v={adaptive_v_spacing:.2f}, complexity_factor={complexity_factor:.2f}, integrators={integrator_count}")

    def _adjust_font_size(self):
                                                                            
        
                                                 
        total_components = 0
        total_measurements = 0
        
        for i in range(self.m-1):
            for j in range(self.n):
                if self.has_vedge[i][j] and self.vcomp_type[i][j] != TYPE_OPEN:
                    total_components += 1
                    if self.vcomp_measure[i][j] != MEAS_TYPE_NONE:
                        total_measurements += 1
                        
        for i in range(self.m):
            for j in range(self.n-1):
                if self.has_hedge[i][j] and self.hcomp_type[i][j] != TYPE_OPEN:
                    total_components += 1
                    if self.hcomp_measure[i][j] != MEAS_TYPE_NONE:
                        total_measurements += 1
        
                           
        grid_area = self.m * self.n
        component_density = total_components / max(grid_area, 1)
        measurement_density = total_measurements / max(total_components, 1)
        
                                           
        if component_density > 0.7 or measurement_density > 0.4:
            self.latex_font_size = "\\small"
        elif component_density > 0.5 or measurement_density > 0.3:
            self.latex_font_size = "\\normalsize"
        elif component_density > 0.3 or measurement_density > 0.2:
            self.latex_font_size = "\\large"
        else:
            self.latex_font_size = "\\Large"
            
        print(f"Font size adjusted to: {self.latex_font_size}, component_density={component_density:.2f}, measurement_density={measurement_density:.2f}")

    def _optimize_measurement_placement(self):
                                                                                                   
        
                                                                             
        measurement_map = np.zeros((self.m, self.n), dtype=int)                                
        controlled_source_map = np.zeros((self.m, self.n), dtype=int)                                   
        component_complexity_map = np.zeros((self.m, self.n), dtype=float)                              
        
                                                                                 
        for i in range(self.m-1):
            for j in range(self.n):
                if self.has_vedge[i][j]:
                    component_complexity = 0
                    if self.vcomp_measure[i][j] != MEAS_TYPE_NONE:
                        measurement_map[i][j] = self.vcomp_measure[i][j]
                        measurement_map[i+1][j] = self.vcomp_measure[i][j]                       
                        component_complexity += 0.5
                    if self.vcomp_type[i][j] in [TYPE_VCCS, TYPE_VCVS, TYPE_CCCS, TYPE_CCVS]:
                        controlled_source_map[i][j] = 1
                        controlled_source_map[i+1][j] = 1                       
                        component_complexity += 1.0                                          
                    if self.vcomp_type[i][j] in [TYPE_VOLTAGE_SOURCE, TYPE_CURRENT_SOURCE]:
                        component_complexity += 0.3                                       
                    
                    component_complexity_map[i][j] += component_complexity
                    component_complexity_map[i+1][j] += component_complexity
                        
        for i in range(self.m):
            for j in range(self.n-1):
                if self.has_hedge[i][j]:
                    component_complexity = 0
                    if self.hcomp_measure[i][j] != MEAS_TYPE_NONE:
                        measurement_map[i][j] = self.hcomp_measure[i][j]
                        measurement_map[i][j+1] = self.hcomp_measure[i][j]                       
                        component_complexity += 0.5
                    if self.hcomp_type[i][j] in [TYPE_VCCS, TYPE_VCVS, TYPE_CCCS, TYPE_CCVS]:
                        controlled_source_map[i][j] = 1
                        controlled_source_map[i][j+1] = 1                       
                        component_complexity += 1.0                                          
                    if self.hcomp_type[i][j] in [TYPE_VOLTAGE_SOURCE, TYPE_CURRENT_SOURCE]:
                        component_complexity += 0.3                                       
                    
                    component_complexity_map[i][j] += component_complexity
                    component_complexity_map[i][j+1] += component_complexity
        
                                                                                  
        conflict_resolved = 0
        high_conflict_areas = []
        
        for i in range(self.m):
            for j in range(self.n):
                                                                                           
                local_measurements = 0
                local_controlled = 0
                local_complexity = 0
                
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.m and 0 <= nj < self.n:
                            weight = 1.0 / (abs(di) + abs(dj) + 1)                     
                            if measurement_map[ni][nj] > 0:
                                local_measurements += weight
                            if controlled_source_map[ni][nj] > 0:
                                local_controlled += weight
                            local_complexity += component_complexity_map[ni][nj] * weight
                
                                                                          
                if (local_measurements > 1.5 and local_controlled > 0.5) or local_complexity > 2.0:
                    high_conflict_areas.append((i, j, local_complexity))
        
                                                                        
        high_conflict_areas.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, complexity in high_conflict_areas[:10]:                                      
                                                   
            
                                                                                  
            edges_to_check = []
            if i > 0 and j < self.n:
                edges_to_check.append(('v', i-1, j))
            if i < self.m-1 and j < self.n:
                edges_to_check.append(('v', i, j))
            if j > 0 and i < self.m:
                edges_to_check.append(('h', i, j-1))
            if j < self.n-1 and i < self.m:
                edges_to_check.append(('h', i, j))
            
            for edge_type, ei, ej in edges_to_check:
                if edge_type == 'v' and self.has_vedge[ei][ej]:
                                                                                                     
                    if (self.vcomp_type[ei][ej] in [TYPE_VCCS, TYPE_CCCS] and 
                        self.vcomp_measure[ei][ej] == MEAS_TYPE_CURRENT):
                        self.vcomp_measure[ei][ej] = MEAS_TYPE_NONE
                        self.vcomp_measure_label[ei][ej] = -1
                        conflict_resolved += 1
                        print(f"Removed current measurement from controlled source at vedge ({ei},{ej})")
                elif edge_type == 'h' and self.has_hedge[ei][ej]:
                                                                                                     
                    if (self.hcomp_type[ei][ej] in [TYPE_VCCS, TYPE_CCCS] and 
                        self.hcomp_measure[ei][ej] == MEAS_TYPE_CURRENT):
                        self.hcomp_measure[ei][ej] = MEAS_TYPE_NONE
                        self.hcomp_measure_label[ei][ej] = -1
                        conflict_resolved += 1
                        print(f"Removed current measurement from controlled source at hedge ({ei},{ej})")
        
        print(f"Advanced measurement placement optimized: {conflict_resolved} potential conflicts resolved")
        print(f"Identified {len(high_conflict_areas)} high-conflict areas")

def gen_circuit(note="v1", id="", symbolic=False, simple_circuits=False, integrator=False, rlc=False, no_meas=False):

                       
    if int(note[1:]) <= 9:
        raise NotImplementedError
    
          
    elif int(note[1:]) == 10:

        num_edge_choices = [2]*3 + [3]*5 + [4]*3 + [5]*2 + [6]*1 + [7]*1 + [8]*1
        num_source_choices = [TYPE_VOLTAGE_SOURCE]*5 + [TYPE_CURRENT_SOURCE]*5 + [TYPE_VCCS]*2 + [TYPE_VCVS]*2 + [TYPE_CCCS]*2 + [TYPE_CCVS]*2

        m = np.random.choice(num_edge_choices)
                                          
        n = np.random.choice(num_edge_choices)
        vertical_dis = np.arange(m)* 3 + np.random.uniform(-0.5, 0.5, size=(m,))
        horizontal_dis = np.arange(n)* 3 + np.random.uniform(-0.5, 0.5, size=(n,))

        num_short_max = 0
                                   
        cut_outer_edge_rate = 1
        cut_corner_rate = 0.2

        cut_left_top = random.random()<cut_corner_rate
        cut_left_bottom = random.random()<cut_corner_rate
        cut_right_top = random.random()<cut_corner_rate
        cut_right_bottom = random.random()<cut_corner_rate
        while num_short_max < 1:
            has_vedge = np.random.randint(0, 2, size=(m-1, n))         
            has_hedge = np.random.randint(0, 2, size=(m, n-1))         

            for i in range(m-1):
                has_vedge[i][0] = int(random.random() < cut_outer_edge_rate)
                has_vedge[i][n-1] = int(random.random() < cut_outer_edge_rate)
                                                         
            has_hedge = np.random.randint(0, 2, size=(m, n-1))         
            for j in range(n-1):
                has_hedge[0][j] = int(random.random() < cut_outer_edge_rate)
                has_hedge[m-1][j] = int(random.random() < cut_outer_edge_rate)
                                                         
            
            num_edges = np.sum(has_vedge) + np.sum(has_hedge)
            if num_edges > 8:
                if cut_left_bottom:
                    has_vedge[0][0] = 0
                    has_hedge[0][0] = 0
                if cut_left_top:
                    has_vedge[m-2][0] = 0
                    has_hedge[m-1][0] = 0
                if cut_right_bottom:
                    has_vedge[0][n-1] = 0
                    has_hedge[0][n-2] = 0
                if cut_right_top:
                    has_vedge[m-2][n-1] = 0
                    has_hedge[m-1][n-2] = 0

            idxs_has_vedge = np.where(has_vedge == 1)
            idxs_has_vedge = list(zip(idxs_has_vedge[0], idxs_has_vedge[1]))
            idxs_has_hedge = np.where(has_hedge == 1)
            idxs_has_hedge = list(zip(idxs_has_hedge[0], idxs_has_hedge[1]))
            idxs_edge = [(0, i, j) for i, j in idxs_has_vedge] + [(1, i, j) for i, j in idxs_has_hedge]
            
            num_edges = len(idxs_has_vedge) + len(idxs_has_hedge)
            max_num_source = max(min(5, num_edges // 2 - 1), 1)
            num_sources = np.random.randint(1, max_num_source+1)
            sources = np.random.choice(num_source_choices, num_sources)

            num_volsrs = np.sum(sources == TYPE_VOLTAGE_SOURCE)
            num_cursrs = np.sum(sources == TYPE_CURRENT_SOURCE)
            num_vccs = np.sum(sources == TYPE_VCCS)
            num_vcvs = np.sum(sources == TYPE_VCVS)
            num_cccs = np.sum(sources == TYPE_CCCS)
            num_ccvs = np.sum(sources == TYPE_CCVS)

            num_short_max = (num_edges - num_sources) - 2

        print(f"num_short_max: {num_short_max}")
        print(idxs_edge)
        num_short = np.random.randint(0, num_short_max+1)
        num_open = np.random.randint(0, num_short // 2) if num_short > 2 else 0
        
                                                                                  
        remaining_edges = num_edges - num_sources - num_short - num_open
        
                                                                      
        num_small_signal_choices = [0]*2 + [1]*5 + [2]*3                                              
        if not simple_circuits:                                                        
            max_small_signal = min(2, remaining_edges // 2)                               
            num_small_signal = min(np.random.choice(num_small_signal_choices), max_small_signal)
        else:
            num_small_signal = 0
        
                                                             
        if num_small_signal > 0:
            small_signal_types = np.random.choice([TYPE_BJT_SMALL_SIGNAL, TYPE_MOSFET_SMALL_SIGNAL], 
                                                 num_small_signal, p=[0.6, 0.4])                      
            num_bjt_small = np.sum(small_signal_types == TYPE_BJT_SMALL_SIGNAL)
            num_mosfet_small = np.sum(small_signal_types == TYPE_MOSFET_SMALL_SIGNAL)
        else:
            num_bjt_small = 0
            num_mosfet_small = 0
            small_signal_types = []
        
        num_r = remaining_edges - num_small_signal                                    

        np.random.shuffle(idxs_edge)
        idxs_volsrc = idxs_edge[:num_volsrs]
        idxs_cursrc = idxs_edge[num_volsrs:num_volsrs+num_cursrs]
        idxs_vccs = idxs_edge[num_volsrs+num_cursrs:num_volsrs+num_cursrs+num_vccs]
        idxs_vcvs = idxs_edge[num_volsrs+num_cursrs+num_vccs:num_volsrs+num_cursrs+num_vccs+num_vcvs]
        idxs_cccs = idxs_edge[num_volsrs+num_cursrs+num_vccs+num_vcvs:num_volsrs+num_cursrs+num_vccs+num_vcvs+num_cccs]
        idxs_ccvs = idxs_edge[num_volsrs+num_cursrs+num_vccs+num_vcvs+num_cccs:num_sources]
        idxs_r = idxs_edge[num_sources:num_sources+num_r]
        idxs_small_signal = idxs_edge[num_sources+num_r:num_sources+num_r+num_small_signal]
        idxs_open = idxs_edge[num_sources+num_r+num_small_signal:num_sources+num_r+num_small_signal+num_open]

        label_volsrc = np.random.permutation(range(num_volsrs)) + 1
        label_cursrc = np.random.permutation(range(num_cursrs)) + 1
        label_vccs = np.random.permutation(range(num_vccs)) + 1
        label_vcvs = np.random.permutation(range(num_vcvs)) + 1
        label_cccs = np.random.permutation(range(num_cccs)) + 1
        label_ccvs = np.random.permutation(range(num_ccvs)) + 1

        label_r = np.random.permutation(range(num_r)) + 1
        label_bjt_small = np.random.permutation(range(num_bjt_small)) + 1 if num_bjt_small > 0 else []
        label_mosfet_small = np.random.permutation(range(num_mosfet_small)) + 1 if num_mosfet_small > 0 else []

        vcomp_type = np.zeros((m-1, n))
        hcomp_type = np.zeros((m, n-1))
        vcomp_label = np.zeros((m-1, n))
        hcomp_label = np.zeros((m, n-1))
        vcomp_value = np.zeros((m-1, n))
        hcomp_value = np.zeros((m, n-1))

        vcomp_value_unit = np.zeros((m-1, n))
        hcomp_value_unit = np.zeros((m, n-1))

        vcomp_direction = np.random.randint(0, 2, size=(m-1, n))         
        hcomp_direction = np.random.randint(0, 2, size=(m, n-1))         

        vcomp_measure = np.zeros((m-1, n))
        hcomp_measure = np.zeros((m, n-1))

        vcomp_measure_label = np.zeros((m-1, n))
        hcomp_measure_label = np.zeros((m, n-1))

        vcomp_measure_direction = np.random.randint(0, 2, size=(m-1, n))         
        hcomp_measure_direction = np.random.randint(0, 2, size=(m, n-1))         

        vcomp_control_meas_label = np.zeros((m-1, n))   
        hcomp_control_meas_label = np.zeros((m, n-1))
        
        min_value_r, max_value_r = 1, 100
        min_value_v, max_value_v = 1, 100
        min_value_i, max_value_i = 1, 100

                          
        num_measure_choices = list(range(0, num_r+1)) + [0]*5+[1]*5+[2]*2
        num_measure = np.random.choice(num_measure_choices)
        if num_measure > 0:
            num_measure_i = np.random.randint(0, num_measure+1)
            num_measure_v = num_measure - num_measure_i
        else:
            num_measure_i = 0
            num_measure_v = 0
        if num_measure_i < num_cccs + num_ccvs:
            num_measure_i = num_cccs + num_ccvs
        if num_measure_v < num_vccs + num_vcvs:
            num_measure_v = num_vccs + num_vcvs
        num_measure = num_measure_i + num_measure_v

        measure_label_sets = np.random.choice(range(-1, 100), num_measure, replace=False)
        
        idxs_measure_i = random.sample(idxs_edge, num_measure_i)
        idxs_measure_v = random.sample(list(set(idxs_edge) - set(idxs_measure_i)) + (idxs_cursrc), num_measure_v)
    
        for l, (s, i, j) in enumerate(idxs_measure_i):
            if s == 0:
                vcomp_measure[i][j] = MEAS_TYPE_CURRENT
                vcomp_measure_label[i][j] = measure_label_sets[l]
            else:
                hcomp_measure[i][j] = MEAS_TYPE_CURRENT
                hcomp_measure_label[i][j] = measure_label_sets[l]
        for l, (s, i, j) in enumerate(idxs_measure_v):
            if s == 0:
                vcomp_measure[i][j] = MEAS_TYPE_VOLTAGE
                vcomp_measure_label[i][j] = measure_label_sets[l]
            else:
                hcomp_measure[i][j] = MEAS_TYPE_VOLTAGE
                hcomp_measure_label[i][j] = measure_label_sets[l]


        for l, (s, i, j) in enumerate(idxs_open):
            if s == 0:
                vcomp_type[i][j] = TYPE_OPEN
            else:
                hcomp_type[i][j] = TYPE_OPEN

        for l, (s, i, j) in enumerate(idxs_volsrc):
            if s == 0:
                vcomp_type[i][j] = TYPE_VOLTAGE_SOURCE
                vcomp_label[i][j] = label_volsrc[l]
                vcomp_value[i][j] = np.random.randint(min_value_v, max_value_v)
            else:
                hcomp_type[i][j] = TYPE_VOLTAGE_SOURCE
                hcomp_label[i][j] = label_volsrc[l]
                hcomp_value[i][j] = np.random.randint(min_value_v, max_value_v)
        for l, (s, i, j) in enumerate(idxs_cursrc):
            if s == 0:
                vcomp_type[i][j] = TYPE_CURRENT_SOURCE
                vcomp_label[i][j] = label_cursrc[l]
                vcomp_value[i][j] = np.random.randint(min_value_i, max_value_i)
            else:
                hcomp_type[i][j] = TYPE_CURRENT_SOURCE
                hcomp_label[i][j] = label_cursrc[l]
                hcomp_value[i][j] = np.random.randint(min_value_i, max_value_i)
        for l, (s, i, j) in enumerate(idxs_r):
            if s == 0:
                vcomp_type[i][j] = TYPE_RESISTOR
                vcomp_label[i][j] = label_r[l]
                vcomp_value[i][j] = np.random.randint(min_value_r, max_value_r)
            else:
                hcomp_type[i][j] = TYPE_RESISTOR
                hcomp_label[i][j] = label_r[l]
                hcomp_value[i][j] = np.random.randint(min_value_r, max_value_r)
        for l, (s, i, j) in enumerate(idxs_vccs):
            if s == 0:
                vcomp_type[i][j] = TYPE_VCCS
                vcomp_label[i][j] = label_vccs[l]
                vcomp_value[i][j] = np.random.randint(min_value_v, max_value_v)       
            else:
                hcomp_type[i][j] = TYPE_VCCS
                hcomp_label[i][j] = label_vccs[l]
                hcomp_value[i][j] = np.random.randint(min_value_v, max_value_v)
        for l, (s, i, j) in enumerate(idxs_vcvs):
            if s == 0:
                vcomp_type[i][j] = TYPE_VCVS
                vcomp_label[i][j] = label_vcvs[l]
                vcomp_value[i][j] = np.random.randint(min_value_v, max_value_v)
            else:
                hcomp_type[i][j] = TYPE_VCVS
                hcomp_label[i][j] = label_vcvs[l]
                hcomp_value[i][j] = np.random.randint(min_value_v, max_value_v)
        for l, (s, i, j) in enumerate(idxs_cccs):
            if s == 0:
                vcomp_type[i][j] = TYPE_CCCS
                vcomp_label[i][j] = label_cccs[l]
                vcomp_value[i][j] = np.random.randint(min_value_i, max_value_i)
            else:
                hcomp_type[i][j] = TYPE_CCCS
                hcomp_label[i][j] = label_cccs[l]
                hcomp_value[i][j] = np.random.randint(min_value_i, max_value_i)
        for l, (s, i, j) in enumerate(idxs_ccvs):
            if s == 0:
                vcomp_type[i][j] = TYPE_CCVS
                vcomp_label[i][j] = label_ccvs[l]
                vcomp_value[i][j] = np.random.randint(min_value_i, max_value_i)
            else:
                hcomp_type[i][j] = TYPE_CCVS
                hcomp_label[i][j] = label_ccvs[l]
                hcomp_value[i][j] = np.random.randint(min_value_i, max_value_i)

               
        for l, (s,i,j) in enumerate(idxs_vccs + idxs_vcvs):
            if s == 0:
                control_measure_voltage_idx = random.choice(idxs_measure_v)
                if control_measure_voltage_idx[0] == 0:
                    vcomp_control_meas_label[i][j] = vcomp_measure_label[control_measure_voltage_idx[1]][control_measure_voltage_idx[2]]
                elif control_measure_voltage_idx[0] == 1:
                    vcomp_control_meas_label[i][j] = hcomp_measure_label[control_measure_voltage_idx[1]][control_measure_voltage_idx[2]]
            else:
                control_measure_voltage_idx = random.choice(idxs_measure_v)
                if control_measure_voltage_idx[0] == 0:
                    hcomp_control_meas_label[i][j] = vcomp_measure_label[control_measure_voltage_idx[1]][control_measure_voltage_idx[2]]
                elif control_measure_voltage_idx[0] == 1:
                    hcomp_control_meas_label[i][j] = hcomp_measure_label[control_measure_voltage_idx[1]][control_measure_voltage_idx[2]]
        for l, (s,i,j) in enumerate(idxs_cccs + idxs_ccvs):
            if s == 0:
                control_measure_current_idx = random.choice(idxs_measure_i)
                if control_measure_current_idx[0] == 0:
                    vcomp_control_meas_label[i][j] = vcomp_measure_label[control_measure_current_idx[1]][control_measure_current_idx[2]]
                elif control_measure_current_idx[0] == 1:
                    vcomp_control_meas_label[i][j] = hcomp_measure_label[control_measure_current_idx[1]][control_measure_current_idx[2]]
            else:
                control_measure_current_idx = random.choice(idxs_measure_i)
                if control_measure_current_idx[0] == 0:
                    hcomp_control_meas_label[i][j] = vcomp_measure_label[control_measure_current_idx[1]][control_measure_current_idx[2]]
                elif control_measure_current_idx[0] == 1:
                    hcomp_control_meas_label[i][j] = hcomp_measure_label[control_measure_current_idx[1]][control_measure_current_idx[2]]

                                               
        bjt_label_idx = 0
        mosfet_label_idx = 0
        for l, (s, i, j) in enumerate(idxs_small_signal):
            if l < len(small_signal_types):
                component_type = small_signal_types[l]
                if s == 0:                      
                    vcomp_type[i][j] = component_type
                    if component_type == TYPE_BJT_SMALL_SIGNAL:
                        vcomp_label[i][j] = label_bjt_small[bjt_label_idx]
                        vcomp_value[i][j] = np.random.randint(10, 100)                  
                        bjt_label_idx += 1
                    elif component_type == TYPE_MOSFET_SMALL_SIGNAL:
                        vcomp_label[i][j] = label_mosfet_small[mosfet_label_idx]
                        vcomp_value[i][j] = np.random.randint(5, 50)                  
                        mosfet_label_idx += 1
                else:                        
                    hcomp_type[i][j] = component_type
                    if component_type == TYPE_BJT_SMALL_SIGNAL:
                        hcomp_label[i][j] = label_bjt_small[bjt_label_idx]
                        hcomp_value[i][j] = np.random.randint(10, 100)                  
                        bjt_label_idx += 1
                    elif component_type == TYPE_MOSFET_SMALL_SIGNAL:
                        hcomp_label[i][j] = label_mosfet_small[mosfet_label_idx]
                        hcomp_value[i][j] = np.random.randint(5, 50)                  
                        mosfet_label_idx += 1

        print(f"vcomp_value: {vcomp_value}\n\nhcomp_value: {hcomp_value}")
        print(f"vcomp_value_unit: {vcomp_value_unit}\n\nhcomp_value_unit: {hcomp_value_unit}")
        print(f"Small signal models added: BJT={num_bjt_small}, MOSFET={num_mosfet_small}")
        print(f"Debug: simple_circuits={simple_circuits}, remaining_edges={remaining_edges}, max_small_signal={max_small_signal if not simple_circuits else 'N/A'}, num_small_signal={num_small_signal}")

                                                                       
        unit_choices = [UNIT_MODE_1]                                   
        vcomp_value_unit = np.random.choice(unit_choices, size=(m-1, n))
        hcomp_value_unit = np.random.choice(unit_choices, size=(m, n-1))
        
                                      
        use_value_annotation = bool(random.getrandbits(1))
                                                                                   
        label_str_subscript = False
        label_numerical_subscript = not label_str_subscript

                                   
        vcomp_type = vcomp_type.astype(int)
        hcomp_type = hcomp_type.astype(int)
        vcomp_label = vcomp_label.astype(int)
        hcomp_label = hcomp_label.astype(int)
        vcomp_value = vcomp_value.astype(int)
        hcomp_value = hcomp_value.astype(int)
        vcomp_value_unit = vcomp_value_unit.astype(int)
        hcomp_value_unit = hcomp_value_unit.astype(int)

        vcomp_measure = vcomp_measure.astype(int)
        hcomp_measure = hcomp_measure.astype(int)
        vcomp_measure_label = vcomp_measure_label.astype(int)
        hcomp_measure_label = hcomp_measure_label.astype(int)
        vcomp_measure_direction = vcomp_measure_direction.astype(int)
        hcomp_measure_direction = hcomp_measure_direction.astype(int)
        vcomp_control_meas_label = vcomp_control_meas_label.astype(int)
        hcomp_control_meas_label = hcomp_control_meas_label.astype(int)

        print("#"*100)
        print("Generate a random grid for circuit ... ")
        print(f"has_vedge: {has_vedge}\n\nhas_hedge: {has_hedge}")
        print(f"vertical_dis: {vertical_dis}\n\nhorizontal_dis: {horizontal_dis}")
        print(f"m:{m}, n:{n}\n\nnum_edges:{num_edges},\nnum_sources: {num_sources},\nnum_volsrs: {num_volsrs},\nnum_cursrs: {num_cursrs}\nnum_resistors: {num_r}")
        print(f"use_value_annotation: {use_value_annotation}\nlabel_numerical_subscript: {label_numerical_subscript}")

        print(f"vcomp_type: {vcomp_type}\n\nhcomp_type: {hcomp_type}")
        print(f"vcomp_label: {vcomp_label}\n\nhcomp_label: {hcomp_label}")
        print(f"vcomp_value: {vcomp_value}\n\nhcomp_value: {hcomp_value}")
        print(f"vcomp_value_unit: {vcomp_value_unit}\n\nhcomp_value_unit: {hcomp_value_unit}")
        print(f"vcomp_measure: {vcomp_measure}\n\nhcomp_measure: {hcomp_measure}")
        print(f"vcomp_measure_label: {vcomp_measure_label}\n\nhcomp_measure_label: {hcomp_measure_label}")
        print(f"vcomp_measure_direction: {vcomp_measure_direction}\n\nhcomp_measure_direction: {hcomp_measure_direction}")
        print(f"vcomp_control_meas_label: {vcomp_control_meas_label}\n\nhcomp_control_meas_label: {hcomp_control_meas_label}")

                                                                            
                                                                            
                                          
                                                                              
                                                   
                                                                              
                                                                             
                                                                             
                                 
                                                                            

                                           
        for ii in range(m-1):
            for jj in range(n):
                if vcomp_type[ii][jj] == TYPE_CURRENT_SOURCE:
                    vcomp_type[ii][jj] = TYPE_RESISTOR
                    vcomp_value[ii][jj] = np.random.randint(min_value_r, max_value_r)
                    print(f"Converted current source at vedge ({ii},{jj}) to resistor")
        for ii in range(m):
            for jj in range(n-1):
                if hcomp_type[ii][jj] == TYPE_CURRENT_SOURCE:
                    hcomp_type[ii][jj] = TYPE_RESISTOR
                    hcomp_value[ii][jj] = np.random.randint(min_value_r, max_value_r)
                    print(f"Converted current source at hedge ({ii},{jj}) to resistor")

                                                  
        voltage_positions = []                                  
        for ii in range(m-1):
            for jj in range(n):
                if vcomp_type[ii][jj] == TYPE_VOLTAGE_SOURCE:
                    voltage_positions.append(('v', ii, jj))
        for ii in range(m):
            for jj in range(n-1):
                if hcomp_type[ii][jj] == TYPE_VOLTAGE_SOURCE:
                    voltage_positions.append(('h', ii, jj))

        if len(voltage_positions) == 0:
                                                                
            candidate_edges = [('v', ii, jj) for ii in range(m-1) for jj in range(n) if has_vedge[ii][jj] and vcomp_type[ii][jj] != TYPE_OPEN] + \
                              [('h', ii, jj) for ii in range(m) for jj in range(n-1) if has_hedge[ii][jj] and hcomp_type[ii][jj] != TYPE_OPEN]
            if candidate_edges:
                chosen_edge = random.choice(candidate_edges)
                if chosen_edge[0] == 'v':
                    ii, jj = chosen_edge[1], chosen_edge[2]
                    vcomp_type[ii][jj] = TYPE_VOLTAGE_SOURCE
                    vcomp_value[ii][jj] = np.random.randint(min_value_v, max_value_v)
                    print(f"Promoted edge at vedge ({ii},{jj}) to voltage source")
                else:
                    ii, jj = chosen_edge[1], chosen_edge[2]
                    hcomp_type[ii][jj] = TYPE_VOLTAGE_SOURCE
                    hcomp_value[ii][jj] = np.random.randint(min_value_v, max_value_v)
                    print(f"Promoted edge at hedge ({ii},{jj}) to voltage source")
        elif len(voltage_positions) > 1:
                                                                            
            keep = voltage_positions[0]
            for pos in voltage_positions[1:]:
                if pos[0] == 'v':
                    vcomp_type[pos[1]][pos[2]] = TYPE_RESISTOR
                    vcomp_value[pos[1]][pos[2]] = np.random.randint(min_value_r, max_value_r)
                    print(f"Demoted extra voltage source at vedge ({pos[1]},{pos[2]}) to resistor")
                else:
                    hcomp_type[pos[1]][pos[2]] = TYPE_RESISTOR
                    hcomp_value[pos[1]][pos[2]] = np.random.randint(min_value_r, max_value_r)
                    print(f"Demoted extra voltage source at hedge ({pos[1]},{pos[2]}) to resistor")

        print(f"Voltage source constraint enforced: {len(voltage_positions)} initial voltage sources found")

                                                                                 
        reassign_unique_labels(vcomp_type, hcomp_type, vcomp_label, hcomp_label, m, n)
        print("Component labels reassigned to ensure uniqueness")

                                                                            
                                                                    
                                                                            
        if integrator:
                                                  
            integrator_positions = []                                  
            for ii in range(m-1):
                for jj in range(n):
                    if vcomp_type[ii][jj] == TYPE_OPAMP_INTEGRATOR:
                        integrator_positions.append(('v', ii, jj))
            for ii in range(m):
                for jj in range(n-1):
                    if hcomp_type[ii][jj] == TYPE_OPAMP_INTEGRATOR:
                        integrator_positions.append(('h', ii, jj))

            if len(integrator_positions) == 0:
                                                                     
                candidate_edges = [('v', ii, jj) for ii in range(m-1) for jj in range(n) if has_vedge[ii][jj] and vcomp_type[ii][jj] == TYPE_RESISTOR] + \
                                  [('h', ii, jj) for ii in range(m) for jj in range(n-1) if has_hedge[ii][jj] and hcomp_type[ii][jj] == TYPE_RESISTOR]
                if candidate_edges:
                    chosen_edge = random.choice(candidate_edges)
                    if chosen_edge[0] == 'v':
                        ii, jj = chosen_edge[1], chosen_edge[2]
                        vcomp_type[ii][jj] = TYPE_OPAMP_INTEGRATOR
                        vcomp_value[ii][jj] = np.random.randint(min_value_r, max_value_r)                        
                        print(f"Promoted resistor at vedge ({ii},{jj}) to integrator")
                    else:
                        ii, jj = chosen_edge[1], chosen_edge[2]
                        hcomp_type[ii][jj] = TYPE_OPAMP_INTEGRATOR
                        hcomp_value[ii][jj] = np.random.randint(min_value_r, max_value_r)                        
                        print(f"Promoted resistor at hedge ({ii},{jj}) to integrator")
            elif len(integrator_positions) > 1:
                                                                            
                keep = integrator_positions[0]
                for pos in integrator_positions[1:]:
                    if pos[0] == 'v':
                        vcomp_type[pos[1]][pos[2]] = TYPE_RESISTOR
                        vcomp_value[pos[1]][pos[2]] = np.random.randint(min_value_r, max_value_r)
                        print(f"Demoted extra integrator at vedge ({pos[1]},{pos[2]}) to resistor")
                    else:
                        hcomp_type[pos[1]][pos[2]] = TYPE_RESISTOR
                        hcomp_value[pos[1]][pos[2]] = np.random.randint(min_value_r, max_value_r)
                        print(f"Demoted extra integrator at hedge ({pos[1]},{pos[2]}) to resistor")

            print(f"Integrator constraint enforced: {len(integrator_positions)} initial integrators found")

                                                                       
            reassign_unique_labels(vcomp_type, hcomp_type, vcomp_label, hcomp_label, m, n)
            print("Component labels reassigned after integrator enforcement")

                                                                                                                                                     
        circ = Circuit( m=m, n=n, \
                        vertical_dis=vertical_dis, horizontal_dis=horizontal_dis, \
                        has_vedge=has_vedge, has_hedge=has_hedge, \
                        vcomp_type=vcomp_type, hcomp_type=hcomp_type, \
                        vcomp_label=vcomp_label, hcomp_label=hcomp_label, \
                        vcomp_value=vcomp_value, hcomp_value=hcomp_value, \
                        vcomp_value_unit=vcomp_value_unit, hcomp_value_unit=hcomp_value_unit, \
                        vcomp_measure=vcomp_measure, hcomp_measure=hcomp_measure, \
                        vcomp_measure_label=vcomp_measure_label, hcomp_measure_label=hcomp_measure_label, \
                        use_value_annotation=use_value_annotation, note=note, id=id,
                        vcomp_direction=vcomp_direction, hcomp_direction=hcomp_direction,
                        vcomp_measure_direction=vcomp_measure_direction, hcomp_measure_direction=hcomp_measure_direction,
                        vcomp_control_meas_label=vcomp_control_meas_label, hcomp_control_meas_label=hcomp_control_meas_label,
                        label_numerical_subscript=label_numerical_subscript,
                        rlc=rlc,
                        no_meas=no_meas)                                               
    
    elif int(note[1:]) == 11:

                                            
        if simple_circuits:
                                                           
            num_grid_options = [2, 3, 4]
            num_grid_dis =     [8, 6, 1]                                   
                                                                          
            num_comp_dis = [8, 3, 0, 8, 2, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8]                                                         
            num_comp_dis_outer = [6, 3, 0, 6, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6]                                       
        else:
                                          
            num_grid_options = [2, 3, 4, 5, 6, 7, 8]
            num_grid_dis =     [6, 8, 2, 0, 0, 0, 0]                                                    
                                                                          
            num_comp_dis = [10, 4, 0, 10, 3, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10]                                                         
            num_comp_dis_outer = [8, 4, 0, 8, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8]                                   
        
        num_grid_choices = []
        for op, dis in zip(num_grid_options, num_grid_dis):
            num_grid_choices += [op]*dis
 
                                                                                                                                     
                                                                                                                     
        
        num_comp_dis = [12,  4, 0, 15, 6, 5, 8,    1,    4,    3,    4,    0,        0,            0,      0,          0,             0]  
                                                                                                                                              
        num_comp_dis_outer = [10, 4, 0, 10, 1, 1, 0, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0]                                     

        num_comp_choices = []
        num_comp_choices_outer = []
                                                                                      
        for op, dis in zip(range(18), num_comp_dis):
            num_comp_choices += [op]*dis
        for op, dis in zip(range(18), num_comp_dis_outer):
            num_comp_choices_outer += [op]*dis

        vertical_dis_mean, vertical_dis_std = 4.0, 0.4                                              
        horizontal_dis_mean, horizontal_dis_std = 4.0, 0.4                                              

                                                                                
        comp_mean_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10]                            
        comp_max_value = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 50, 50, 1, 10, 10, 50, 100]                       

                                                                       
        unit_choices = [UNIT_MODE_1]                                   

        meas_dis = [20, 1, 1]                                                            
        meas_choices = [MEAS_TYPE_NONE]*meas_dis[0] + [MEAS_TYPE_VOLTAGE]*meas_dis[1] + [MEAS_TYPE_CURRENT]*meas_dis[2]
        meas_dir_prob = 0.5

        meas_label_choices = range(-1, 10)

        use_value_annotation_prob = 0.9                                                            

                      
        m = np.random.choice(num_grid_choices)
        if m == 4:
            num_grid_choices.remove(4)
        n = np.random.choice(num_grid_choices)
        vertical_dis = np.arange(m)* vertical_dis_mean + np.random.uniform(-vertical_dis_std, vertical_dis_std, size=(m,))
        horizontal_dis = np.arange(n)* horizontal_dis_mean + np.random.uniform(-horizontal_dis_std, horizontal_dis_std, size=(n,))

        while True:

                                     
            has_vedge = np.ones((m-1, n), dtype=int)
            has_hedge = np.ones((m, n-1), dtype=int)

            vcomp_type = np.zeros((m-1, n), dtype=int)
            hcomp_type = np.zeros((m, n-1), dtype=int)
            vcomp_label = np.zeros((m-1, n))
            hcomp_label = np.zeros((m, n-1))
            vcomp_value = np.zeros((m-1, n))
            hcomp_value = np.zeros((m, n-1))

            vcomp_value_unit = np.zeros((m-1, n), dtype=int)
            hcomp_value_unit = np.zeros((m, n-1), dtype=int)

            vcomp_direction = np.zeros((m-1, n), dtype=int)         
            hcomp_direction = np.zeros((m, n-1), dtype=int)         

            vcomp_measure = np.zeros((m-1, n), dtype=int)
            hcomp_measure = np.zeros((m, n-1), dtype=int)

            vcomp_measure_label = np.zeros((m-1, n))
            hcomp_measure_label = np.zeros((m, n-1))

            vcomp_measure_direction = np.zeros((m-1, n), dtype=int)         
            hcomp_measure_direction = np.zeros((m, n-1), dtype=int)         

            vcomp_control_meas_label = np.zeros((m-1, n))   
            hcomp_control_meas_label = np.zeros((m, n-1))

                                
            comp_cnt = [0] * 18                                                 
            meas_label_stat = {
                MEAS_TYPE_NONE: [],
                MEAS_TYPE_VOLTAGE: [],
                MEAS_TYPE_CURRENT: []
            }

                                             
            VC_sources = {'v': [], 'h': []}
            IC_sources = {'v': [], 'h': []}
            print(f"has_vedge: {has_vedge}\n\nhas_hedge: {has_hedge}")

            for i in range(m-1):
                for j in range(n):
                    if j == 0 or j == n-1:
                        vcomp_type[i][j] = np.random.choice(num_comp_choices_outer)
                    else:
                        vcomp_type[i][j] = np.random.choice(num_comp_choices)

                    if vcomp_type[i][j] in [TYPE_VCCS, TYPE_VCVS]:
                        VC_sources["v"].append((i, j))
                    if vcomp_type[i][j] in [TYPE_CCCS, TYPE_CCVS]:
                        IC_sources["v"].append((i, j))
                    if vcomp_type[i][j] == TYPE_OPEN:
                        has_vedge[i][j] = 0
                        continue

                    vcomp_value[i][j] = np.random.randint(comp_mean_value[vcomp_type[i][j]], comp_max_value[vcomp_type[i][j]])
                    vcomp_value_unit[i][j] = np.random.choice(unit_choices)

                    comp_cnt[vcomp_type[i][j]] += 1
                    vcomp_label[i][j] = comp_cnt[vcomp_type[i][j]]

                                                                            
                    proposed_measure = np.random.choice(meas_choices)
                    proposed_label = int(np.random.choice(meas_label_choices))
                    if vcomp_type[i][j] in [TYPE_VCCS, TYPE_VCVS, TYPE_CCCS, TYPE_CCVS]:
                        vcomp_measure[i][j] = MEAS_TYPE_NONE
                        vcomp_measure_label[i][j] = -1
                    else:
                        vcomp_measure[i][j] = proposed_measure
                        vcomp_measure_label[i][j] = proposed_label
                        meas_label_stat[proposed_measure].append(proposed_label)
                    vcomp_direction[i][j] = int(random.random() < meas_dir_prob)

                    print(f"\n\nvcomp_type[{i}][{j}]: {vcomp_type[i][j]}, vcomp_value[{i}][{j}]: {vcomp_value[i][j]}, vcomp_value_unit[{i}][{j}]: {vcomp_value_unit[i][j]}")
                    print(f"vcomp_measure[{i}][{j}]: {vcomp_measure[i][j]}, vcomp_measure_label[{i}][{j}]: {vcomp_measure_label[i][j]}, vcomp_direction[{i}][{j}]: {vcomp_direction[i][j]}")
            for i in range(m):
                for j in range(n-1):
                    if i == 0 or i == m-1:
                        hcomp_type[i][j] = np.random.choice(num_comp_choices_outer)
                    else:
                        hcomp_type[i][j] = np.random.choice(num_comp_choices)

                    if hcomp_type[i][j] in [TYPE_VCCS, TYPE_VCVS]:
                        VC_sources["h"].append((i, j))
                    if hcomp_type[i][j] in [TYPE_CCCS, TYPE_CCVS]:
                        IC_sources["h"].append((i, j))
                    if hcomp_type[i][j] == TYPE_OPEN:
                        has_hedge[i][j] = 0
                        continue
                    
                    hcomp_value[i][j] = np.random.randint(comp_mean_value[hcomp_type[i][j]], comp_max_value[hcomp_type[i][j]])
                    hcomp_value_unit[i][j] = np.random.choice(unit_choices)

                    comp_cnt[hcomp_type[i][j]] += 1
                    hcomp_label[i][j] = comp_cnt[hcomp_type[i][j]]

                                                                            
                    proposed_measure_h = np.random.choice(meas_choices)
                    proposed_label_h = int(np.random.choice(meas_label_choices))
                    if hcomp_type[i][j] in [TYPE_VCCS, TYPE_VCVS, TYPE_CCCS, TYPE_CCVS]:
                        hcomp_measure[i][j] = MEAS_TYPE_NONE
                        hcomp_measure_label[i][j] = -1
                    else:
                        hcomp_measure[i][j] = proposed_measure_h
                        hcomp_measure_label[i][j] = proposed_label_h
                        meas_label_stat[proposed_measure_h].append(proposed_label_h)
                    hcomp_direction[i][j] = int(random.random() < meas_dir_prob)

                    print(f"\n\nhcomp_type[{i}][{j}]: {hcomp_type[i][j]}, hcomp_value[{i}][{j}]: {hcomp_value[i][j]}, hcomp_value_unit[{i}][{j}]: {hcomp_value_unit[i][j]}")
                    print(f"hcomp_measure[{i}][{j}]: {hcomp_measure[i][j]}, hcomp_measure_label[{i}][{j}]: {hcomp_measure_label[i][j]}, hcomp_direction[{i}][{j}]: {hcomp_direction[i][j]}")
            
                                      
            num_vc_sources = len(VC_sources["v"]) + len(VC_sources["h"])
            num_ic_sources = len(IC_sources["v"]) + len(IC_sources["h"])
            num_vmeas = len(meas_label_stat[MEAS_TYPE_VOLTAGE])
            num_imeas = len(meas_label_stat[MEAS_TYPE_CURRENT])

                                                             
            total_dep_sources = num_vc_sources + num_ic_sources
            if total_dep_sources < 1 or total_dep_sources > 2:
                continue

            if (num_vc_sources > 0 and num_vmeas == 0) or (num_ic_sources > 0 and num_imeas == 0):
                continue

            print("VC_sources: ", VC_sources)
            print("IC_sources: ", IC_sources)
            print("meas_label_stat: ", meas_label_stat)

            for i, j in VC_sources["v"]:
                contrl_idx = random.choice(meas_label_stat[MEAS_TYPE_VOLTAGE])
                vcomp_control_meas_label[i][j] = contrl_idx
            for i, j in VC_sources["h"]:
                contrl_idx = random.choice(meas_label_stat[MEAS_TYPE_VOLTAGE])
                hcomp_control_meas_label[i][j] = contrl_idx
            for i, j in IC_sources["v"]:
                contrl_idx = random.choice(meas_label_stat[MEAS_TYPE_CURRENT])
                vcomp_control_meas_label[i][j] = contrl_idx
            for i, j in IC_sources["h"]:
                contrl_idx = random.choice(meas_label_stat[MEAS_TYPE_CURRENT])
                hcomp_control_meas_label[i][j] = contrl_idx
                                                                          
                                                                           
                                                                              
            if rlc:
                                                                  
                has_reactive = False
                for ii in range(m-1):
                    for jj in range(n):
                        if vcomp_type[ii][jj] in [TYPE_CAPACITOR, TYPE_INDUCTOR]:
                            has_reactive = True
                            break
                    if has_reactive:
                        break
                if not has_reactive:
                                                                        
                    candidate_edges = [('v', ii, jj) for ii in range(m-1) for jj in range(n) if has_vedge[ii][jj] and vcomp_type[ii][jj] == TYPE_RESISTOR] + \
                                      [('h', ii, jj) for ii in range(m) for jj in range(n-1) if has_hedge[ii][jj] and hcomp_type[ii][jj] == TYPE_RESISTOR]
                    if candidate_edges:
                        chosen_edge = random.choice(candidate_edges)
                        make_type = random.choice([TYPE_CAPACITOR, TYPE_INDUCTOR])
                        if chosen_edge[0] == 'v':
                            ii, jj = chosen_edge[1], chosen_edge[2]
                            vcomp_type[ii][jj] = make_type
                            vcomp_value[ii][jj] = np.random.randint(comp_mean_value[make_type], comp_max_value[make_type])
                            print(f"Promoted resistor at vedge ({ii},{jj}) to {'C' if make_type==TYPE_CAPACITOR else 'L'} for RLC mode")
                        else:
                            ii, jj = chosen_edge[1], chosen_edge[2]
                            hcomp_type[ii][jj] = make_type
                            hcomp_value[ii][jj] = np.random.randint(comp_mean_value[make_type], comp_max_value[make_type])
                            print(f"Promoted resistor at hedge ({ii},{jj}) to {'C' if make_type==TYPE_CAPACITOR else 'L'} for RLC mode")

            break
        
                                      
        use_value_annotation = bool(random.random() < use_value_annotation_prob)
                                                                                   
        label_str_subscript = False
        label_numerical_subscript = not label_str_subscript

                                   
        vcomp_type = vcomp_type.astype(int)
        hcomp_type = hcomp_type.astype(int)
        vcomp_label = vcomp_label.astype(int)
        hcomp_label = hcomp_label.astype(int)
        vcomp_value = vcomp_value.astype(int)
        hcomp_value = hcomp_value.astype(int)
        vcomp_value_unit = vcomp_value_unit.astype(int)
        hcomp_value_unit = hcomp_value_unit.astype(int)
        vcomp_measure = vcomp_measure.astype(int)
        hcomp_measure = hcomp_measure.astype(int)
        vcomp_measure_label = vcomp_measure_label.astype(int)
        hcomp_measure_label = hcomp_measure_label.astype(int)
        vcomp_measure_direction = vcomp_measure_direction.astype(int)
        hcomp_measure_direction = hcomp_measure_direction.astype(int)
        vcomp_control_meas_label = vcomp_control_meas_label.astype(int)
        hcomp_control_meas_label = hcomp_control_meas_label.astype(int)

        print("#"*100)
        print("Generate a random grid for circuit ... ")
        print(f"has_vedge: {has_vedge}\n\nhas_hedge: {has_hedge}")
        print(f"vertical_dis: {vertical_dis}\n\nhorizontal_dis: {horizontal_dis}")
        print(f"m:{m}, n:{n}\n\ncomp_cnt: {json.dumps(comp_cnt, indent=4)}")
        print(f"use_value_annotation: {use_value_annotation}\nlabel_numerical_subscript: {label_numerical_subscript}")

        print(f"vcomp_type: {vcomp_type}\n\nhcomp_type: {hcomp_type}")
        print(f"vcomp_label: {vcomp_label}\n\nhcomp_label: {hcomp_label}")
        print(f"vcomp_value: {vcomp_value}\n\nhcomp_value: {hcomp_value}")
        print(f"vcomp_value_unit: {vcomp_value_unit}\n\nhcomp_value_unit: {hcomp_value_unit}")
        print(f"vcomp_measure: {vcomp_measure}\n\nhcomp_measure: {hcomp_measure}")
        print(f"vcomp_measure_label: {vcomp_measure_label}\n\nhcomp_measure_label: {hcomp_measure_label}")
        print(f"vcomp_measure_direction: {vcomp_measure_direction}\n\nhcomp_measure_direction: {hcomp_measure_direction}")
        print(f"vcomp_control_meas_label: {vcomp_control_meas_label}\n\nhcomp_control_meas_label: {hcomp_control_meas_label}")

                                                                            
                                                                            
                                          
                                                                              
                                                   
                                                                              
                                                                             
                                                                             
                                 
                                                                            

                                           
        for ii in range(m-1):
            for jj in range(n):
                if vcomp_type[ii][jj] == TYPE_CURRENT_SOURCE:
                    vcomp_type[ii][jj] = TYPE_RESISTOR
                    vcomp_value[ii][jj] = np.random.randint(comp_mean_value[TYPE_RESISTOR], comp_max_value[TYPE_RESISTOR])
                    print(f"Converted current source at vedge ({ii},{jj}) to resistor")
        for ii in range(m):
            for jj in range(n-1):
                if hcomp_type[ii][jj] == TYPE_CURRENT_SOURCE:
                    hcomp_type[ii][jj] = TYPE_RESISTOR
                    hcomp_value[ii][jj] = np.random.randint(comp_mean_value[TYPE_RESISTOR], comp_max_value[TYPE_RESISTOR])
                    print(f"Converted current source at hedge ({ii},{jj}) to resistor")

                                                  
        voltage_positions = []                                  
        for ii in range(m-1):
            for jj in range(n):
                if vcomp_type[ii][jj] == TYPE_VOLTAGE_SOURCE:
                    voltage_positions.append(('v', ii, jj))
        for ii in range(m):
            for jj in range(n-1):
                if hcomp_type[ii][jj] == TYPE_VOLTAGE_SOURCE:
                    voltage_positions.append(('h', ii, jj))

        if len(voltage_positions) == 0:
                                                                
            candidate_edges = [('v', ii, jj) for ii in range(m-1) for jj in range(n) if has_vedge[ii][jj] and vcomp_type[ii][jj] != TYPE_OPEN] + \
                              [('h', ii, jj) for ii in range(m) for jj in range(n-1) if has_hedge[ii][jj] and hcomp_type[ii][jj] != TYPE_OPEN]
            if candidate_edges:
                chosen_edge = random.choice(candidate_edges)
                if chosen_edge[0] == 'v':
                    ii, jj = chosen_edge[1], chosen_edge[2]
                    vcomp_type[ii][jj] = TYPE_VOLTAGE_SOURCE
                    vcomp_value[ii][jj] = np.random.randint(comp_mean_value[TYPE_VOLTAGE_SOURCE], comp_max_value[TYPE_VOLTAGE_SOURCE])
                    print(f"Promoted edge at vedge ({ii},{jj}) to voltage source")
                else:
                    ii, jj = chosen_edge[1], chosen_edge[2]
                    hcomp_type[ii][jj] = TYPE_VOLTAGE_SOURCE
                    hcomp_value[ii][jj] = np.random.randint(comp_mean_value[TYPE_VOLTAGE_SOURCE], comp_max_value[TYPE_VOLTAGE_SOURCE])
                    print(f"Promoted edge at hedge ({ii},{jj}) to voltage source")
        elif len(voltage_positions) > 1:
                                                                            
            keep = voltage_positions[0]
            for pos in voltage_positions[1:]:
                if pos[0] == 'v':
                    vcomp_type[pos[1]][pos[2]] = TYPE_RESISTOR
                    vcomp_value[pos[1]][pos[2]] = np.random.randint(comp_mean_value[TYPE_RESISTOR], comp_max_value[TYPE_RESISTOR])
                    print(f"Demoted extra voltage source at vedge ({pos[1]},{pos[2]}) to resistor")
                else:
                    hcomp_type[pos[1]][pos[2]] = TYPE_RESISTOR
                    hcomp_value[pos[1]][pos[2]] = np.random.randint(comp_mean_value[TYPE_RESISTOR], comp_max_value[TYPE_RESISTOR])
                    print(f"Demoted extra voltage source at hedge ({pos[1]},{pos[2]}) to resistor")

        print(f"Voltage source constraint enforced: {len(voltage_positions)} initial voltage sources found")

                                                                                 
        reassign_unique_labels(vcomp_type, hcomp_type, vcomp_label, hcomp_label, m, n)
        print("Component labels reassigned to ensure uniqueness")

                                                                            
                                                                    
                                                                            
        if integrator:
                                                  
            integrator_positions = []                                  
            for ii in range(m-1):
                for jj in range(n):
                    if vcomp_type[ii][jj] == TYPE_OPAMP_INTEGRATOR:
                        integrator_positions.append(('v', ii, jj))
            for ii in range(m):
                for jj in range(n-1):
                    if hcomp_type[ii][jj] == TYPE_OPAMP_INTEGRATOR:
                        integrator_positions.append(('h', ii, jj))

            if len(integrator_positions) == 0:
                                                                     
                candidate_edges = [('v', ii, jj) for ii in range(m-1) for jj in range(n) if has_vedge[ii][jj] and vcomp_type[ii][jj] == TYPE_RESISTOR] + \
                                  [('h', ii, jj) for ii in range(m) for jj in range(n-1) if has_hedge[ii][jj] and hcomp_type[ii][jj] == TYPE_RESISTOR]
                if candidate_edges:
                    chosen_edge = random.choice(candidate_edges)
                    if chosen_edge[0] == 'v':
                        ii, jj = chosen_edge[1], chosen_edge[2]
                        vcomp_type[ii][jj] = TYPE_OPAMP_INTEGRATOR
                        vcomp_value[ii][jj] = np.random.randint(comp_mean_value[TYPE_RESISTOR], comp_max_value[TYPE_RESISTOR])
                        print(f"Promoted resistor at vedge ({ii},{jj}) to integrator")
                    else:
                        ii, jj = chosen_edge[1], chosen_edge[2]
                        hcomp_type[ii][jj] = TYPE_OPAMP_INTEGRATOR
                        hcomp_value[ii][jj] = np.random.randint(comp_mean_value[TYPE_RESISTOR], comp_max_value[TYPE_RESISTOR])
                        print(f"Promoted resistor at hedge ({ii},{jj}) to integrator")
            elif len(integrator_positions) > 1:
                                                                            
                keep = integrator_positions[0]
                for pos in integrator_positions[1:]:
                    if pos[0] == 'v':
                        vcomp_type[pos[1]][pos[2]] = TYPE_RESISTOR
                        vcomp_value[pos[1]][pos[2]] = np.random.randint(comp_mean_value[TYPE_RESISTOR], comp_max_value[TYPE_RESISTOR])
                        print(f"Demoted extra integrator at vedge ({pos[1]},{pos[2]}) to resistor")
                    else:
                        hcomp_type[pos[1]][pos[2]] = TYPE_RESISTOR
                        hcomp_value[pos[1]][pos[2]] = np.random.randint(comp_mean_value[TYPE_RESISTOR], comp_max_value[TYPE_RESISTOR])
                        print(f"Demoted extra integrator at hedge ({pos[1]},{pos[2]}) to resistor")

            print(f"Integrator constraint enforced: {len(integrator_positions)} initial integrators found")

                                                                       
            reassign_unique_labels(vcomp_type, hcomp_type, vcomp_label, hcomp_label, m, n)
            print("Component labels reassigned after integrator enforcement")

                                                                                                                                                     
                                                                                                                          
        if rlc:
                                                            
            for ii in range(m-1):
                for jj in range(n):
                    if vcomp_type[ii][jj] == TYPE_VOLTAGE_SOURCE:
                        vcomp_measure[ii][jj] = MEAS_TYPE_NONE
            for ii in range(m):
                for jj in range(n-1):
                    if hcomp_type[ii][jj] == TYPE_VOLTAGE_SOURCE:
                        hcomp_measure[ii][jj] = MEAS_TYPE_NONE

        circ = Circuit(m, n, vertical_dis, horizontal_dis, has_vedge, has_hedge, vcomp_type, hcomp_type, vcomp_label, hcomp_label, \
                        vcomp_value=vcomp_value, hcomp_value=hcomp_value, \
                        vcomp_value_unit=vcomp_value_unit, hcomp_value_unit=hcomp_value_unit, \
                        vcomp_measure=vcomp_measure, hcomp_measure=hcomp_measure, \
                        vcomp_measure_label=vcomp_measure_label, hcomp_measure_label=hcomp_measure_label, \
                        use_value_annotation=not symbolic,
                        note=note, id=id,
                        vcomp_direction=vcomp_direction, hcomp_direction=hcomp_direction,
                        vcomp_measure_direction=vcomp_measure_direction, hcomp_measure_direction=hcomp_measure_direction,
                        vcomp_control_meas_label=vcomp_control_meas_label, hcomp_control_meas_label=hcomp_control_meas_label,
                        label_numerical_subscript=label_numerical_subscript,
                        rlc=rlc,
                        no_meas=no_meas)

    else:
        circ = Circuit()
    return circ