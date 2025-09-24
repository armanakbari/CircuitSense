import lcapy as lc
from lcapy import Circuit, s, t, oo, NodalAnalysis, mna
from lcapy import *
from sympy import Eq, Matrix, Symbol, simplify, limit, symbols, MatMul, Wild

                                              
         
                                                                                       
    
                 
                 
                                      
                                                                   
    
              
              
                                                                                           
         
    
                                                                     
                                                                         
                               
                               
                                                               
                                     
                                                                          
                                                                         
                                                   
           
                                                                     
                                                                                 
                                                   
                                              

                      
                      

                                      
                                                                          
                           
                                             
                                                                                                                                  
                                      
                                                 
                                          
                                                                       
                                        
                                                                                         
                                                               

                                
                                                                                  
                                                             
                                               
                               
                                                
                                                                                                                                         
                                           
                                                      
                                               
                                            
                                                                                             
                                                                   

                                                    
                                                                                                          
                                                                      
                                                                                         
                                            

                                              
                                                                                      
                                               
                                             
                                             
                                                  
                                             
                                              
                                        
                                   
                                                                  
                                                       
                                                  
                                             
                                    
                                                      
                                   
                                                                                            

                                                                                                    
                                                                  
                                                                             
                                                       

                                        
                                                                                 
               
                                                                                                                 
    
                                             
                    
                          
                                                                          
                               
                                

                                                                                  
                          
          
                                                 
                                          
                        
                                                                                       
                          
                                                                                       
                       
                              
                               
                                                  
                                                          
    
                                    
                                             
                 
                    
        
                                        
                             
                            
                                        
                                
                                            
                                   
                                             
                       
                                                               
                                                          
                
                                    
        
                              
                    
        
                               
                           
        
                                           
                                                           
                                      
        
                                                 
                                                                
        
                                                 
    
                      

                                          
                                         
          
                         
                                                  
                                 
                          
                       
                          

                 
                       
                              
                              
                                 
                       
                       
                       
                       
                         
                  
                                                   
                           
                                

                                                                                    
      
                                    
                  
                                 
                          
                    
                                                          
                      
                                                          
                 
                                                      
                                            
                                      
                   
          

                                 
                
                 

                                                                                              
                   
                                  
                                    

                  
                                   
                                       
                                 
                                  
                                          
                                                                       
                                                
                                                                           
                      
                                                    
              
                                                    
                                                    
                                                                                           
                      
                           
                  
                                  
                                                   

                                  
                          
                 

                                                        

                                                                                
                        

                                                  


import sympy as sp
from lcapy import s

                 
R1,R2,R3,Rint1 = sp.symbols('R1 R2 R3 Rint1', positive=True)
C1,Cint1       = sp.symbols('C1 Cint1', positive=True)
Ad,V1          = sp.symbols('Ad V1', real=True)

num = (
    Ad*C1*Cint1*R2*R3*Rint1*s**2
    + Ad*Cint1*R2*Rint1*s + Ad*Cint1*R3*Rint1*s
    + C1*Cint1*R2*R3*Rint1*s**2
    + C1*R2*R3*s
    + Cint1*R2*Rint1*s + Cint1*R3*Rint1*s
    + R2 + R3
)

den = s * (
    Ad*C1*Cint1*R1*R2*R3*s**2
    + Ad*C1*Cint1*R1*R2*Rint1*s**2
    + Ad*C1*Cint1*R1*R3*Rint1*s**2
    + Ad*C1*Cint1*R2*R3*Rint1*s**2
    + Ad*C1*R1*R2*s + Ad*Cint1*R1*R2*s + Ad*Cint1*R1*R3*s
    + Ad*Cint1*R2*Rint1*s + Ad*Cint1*R3*Rint1*s
    + C1*Cint1*R1*R2*R3*s**2
    + C1*Cint1*R1*R2*Rint1*s**2
    + C1*Cint1*R1*R3*Rint1*s**2
    + C1*Cint1*R2*R3*Rint1*s**2
    + C1*R1*R2*s + C1*R1*R3*s + C1*R2*R3*s
    + Cint1*R1*R2*s + Cint1*R1*R3*s
    + Cint1*R2*Rint1*s + Cint1*R3*Rint1*s
    + R2 + R3
)

V = V1 * num / den
                                                                          
V_inf = sp.simplify(sp.limit(V, Ad, sp.oo))
print(V_inf)