(* ::Package:: *)

\[Beta]2["B0", 1, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.002344219123474507*Abs[m["e"]]^2*
      Abs[(1.*E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
               "R"]))*m["e"] - 1.3789968659034941*
           E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
               "L"]))*mbar["B0"]*RV["B0"])^2/(mbar["B0"]^4*RV["B0"]^4)] + 
     (0.004457842935155488*
       Abs[(1.*E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                "L"]))*m["e"] - 2.900659239264675*
            E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                "R"]))*mbar["B0"]*RV["B0"])^2/RV["B0"]^2])/Abs[mbar["B0"]]^2
 
\[Beta]2["B0", 1, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 1]]^2*Abs[uR[1, 1]]^2 + 
      Abs[uL[1, 1]]^2*Abs[uR[3, 1]]^2)/2
 
\[Beta]2["B0", 1, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] - Sin[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2*
       Cos[Subscript[\[Theta], 12*"L"]]^2*Cos[Subscript[\[Theta], 13*"L"]]^
        2 + Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 
             12*"L"]]*Cos[Subscript[\[Theta], 23*"L"]]*
           Sin[Subscript[\[Theta], 13*"L"]] - 
          Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 23*"L"]]]^
        2*Cos[Subscript[\[Theta], 12*"R"]]^2*Cos[Subscript[\[Theta], 13*"R"]]^
        2)/2
 
\[Beta]2["B0", 1, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (0.004919033558742541*Abs[m["e"]]^2 + 0.004919033558742541*
       Abs[m["\[Mu]"]]^2)*Abs[1/(mbar["B0"]^2*RV["B0"]^2)]
 
\[Beta]2["B0", 1, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 1]]^2*Abs[uR[1, 2]]^2 + 
      Abs[uL[1, 2]]^2*Abs[uR[3, 1]]^2)/2
 
\[Beta]2["B0", 1, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] - Sin[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2*
       Cos[Subscript[\[Theta], 13*"L"]]^2*Sin[Subscript[\[Theta], 12*"L"]]^
        2 + Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 
             12*"L"]]*Cos[Subscript[\[Theta], 23*"L"]]*
           Sin[Subscript[\[Theta], 13*"L"]] - 
          Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 23*"L"]]]^
        2*Cos[Subscript[\[Theta], 13*"R"]]^2*Sin[Subscript[\[Theta], 12*"R"]]^
        2)/2
 
\[Beta]2["B0", 1, 3, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.014049278093934409*Abs[m["e"]]^2*
      Abs[(1.*m["\[Tau]"] - 2.*mbar["B0"]*RV["B0"])^2/
        (mbar["B0"]^4*RV["B0"]^4)] + (0.056197112375737636*
       Abs[(1.*m["\[Tau]"] - 2.*mbar["B0"]*RV["B0"])^2/RV["B0"]^2])/
      Abs[mbar["B0"]]^2
 
\[Beta]2["B0", 1, 3, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[1, 3]] - (Conjugate[uL[1, 3]]*m["\[Tau]"])/
           (2*mbar["B0"]*RV["B0"])]^2*Abs[uL[3, 1]]^2 + 
      Abs[Conjugate[uL[1, 3]] - (Conjugate[uR[1, 3]]*m["\[Tau]"])/
           (2*mbar["B0"]*RV["B0"])]^2*Abs[uR[3, 1]]^2)/2
 
\[Beta]2["B0", 1, 3, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] - Sin[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[-(m["\[Tau]"]*Sin[Subscript[\[Theta], 13*"L"]])/
           (2*E^(I*Subscript[\[Chi]2, "L"])*mbar["B0"]*RV["B0"]) + 
          Sin[Subscript[\[Theta], 13*"R"]]/E^(I*Subscript[\[Chi]2, "R"])]^2 + 
      Abs[Sin[Subscript[\[Theta], 13*"L"]]/E^(I*Subscript[\[Chi]2, "L"]) - 
          (m["\[Tau]"]*Sin[Subscript[\[Theta], 13*"R"]])/
           (2*E^(I*Subscript[\[Chi]2, "R"])*mbar["B0"]*RV["B0"])]^2*
       Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] - Sin[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2)/2
 
\[Beta]2["B0", 2, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (0.017874611289355375*
       Abs[((1.*E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*m["e"] - 1.3789968659034941*
             E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*mbar["B0"]*RV["B0"])^2*
          (0.3447492164758735*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + 
                Subscript[\[CurlyPhi]1, "L"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*mbar["B0"]*RV["B0"])^2)/RV["B0"]^4] + 
      0.004039891860301862*
       Abs[((0.7251648098161688*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + 
                Subscript[\[CurlyPhi]1, "R"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*mbar["B0"]*RV["B0"])^2*
          (1.*E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*m["e"] - 2.900659239264675*
             E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*mbar["B0"]*RV["B0"])^2)/RV["B0"]^4])/Abs[mbar["B0"]]^4
 
\[Beta]2["B0", 2, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 2]]^2*Abs[uR[1, 1]]^2 + 
      Abs[uL[1, 1]]^2*Abs[uR[3, 2]]^2)/2
 
\[Beta]2["B0", 2, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
           Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] + Cos[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2*
       Cos[Subscript[\[Theta], 12*"L"]]^2*Cos[Subscript[\[Theta], 13*"L"]]^
        2 + Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 
             23*"L"]]*Sin[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 13*"L"]] + 
          Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 23*"L"]]]^
        2*Cos[Subscript[\[Theta], 12*"R"]]^2*Cos[Subscript[\[Theta], 13*"R"]]^
        2)/2
 
\[Beta]2["B0", 2, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.002344219123474507*Abs[m["\[Mu]"]]^2*
      Abs[(1.*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
               "R"]))*m["\[Mu]"] - 1.378996865903494*
           E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
               "L"]))*mbar["B0"]*RV["B0"])^2/(mbar["B0"]^4*RV["B0"]^4)] + 
     (0.004457842935155487*
       Abs[(1.*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                "L"]))*m["\[Mu]"] - 2.9006592392646753*
            E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                "R"]))*mbar["B0"]*RV["B0"])^2/RV["B0"]^2])/Abs[mbar["B0"]]^2
 
\[Beta]2["B0", 2, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 2]]^2*Abs[uR[1, 2]]^2 + 
      Abs[uL[1, 2]]^2*Abs[uR[3, 2]]^2)/2
 
\[Beta]2["B0", 2, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
           Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] + Cos[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2*
       Cos[Subscript[\[Theta], 13*"L"]]^2*Sin[Subscript[\[Theta], 12*"L"]]^
        2 + Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 
             23*"L"]]*Sin[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 13*"L"]] + 
          Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 23*"L"]]]^
        2*Cos[Subscript[\[Theta], 13*"R"]]^2*Sin[Subscript[\[Theta], 12*"R"]]^
        2)/2
 
\[Beta]2["B0", 2, 3, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (0.050928276334906986*Abs[((1.*m["\[Tau]"] - 2.*mbar["B0"]*RV["B0"])^2*
          (0.7251648098161688*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + 
                Subscript[\[CurlyPhi]1, "R"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*mbar["B0"]*RV["B0"])^2)/RV["B0"]^4] + 
      0.10712538871064463*Abs[((1.*m["\[Tau]"] - 2.*mbar["B0"]*RV["B0"])^2*
          (0.3447492164758735*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + 
                Subscript[\[CurlyPhi]1, "L"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*mbar["B0"]*RV["B0"])^2)/RV["B0"]^4])/Abs[mbar["B0"]]^4
 
\[Beta]2["B0", 2, 3, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[1, 3]] - (Conjugate[uL[1, 3]]*m["\[Tau]"])/
           (2*mbar["B0"]*RV["B0"])]^2*Abs[uL[3, 2]]^2 + 
      Abs[Conjugate[uL[1, 3]] - (Conjugate[uR[1, 3]]*m["\[Tau]"])/
           (2*mbar["B0"]*RV["B0"])]^2*Abs[uR[3, 2]]^2)/2
 
\[Beta]2["B0", 2, 3, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
           Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] + Cos[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[-(m["\[Tau]"]*Sin[Subscript[\[Theta], 13*"L"]])/
           (2*E^(I*Subscript[\[Chi]2, "L"])*mbar["B0"]*RV["B0"]) + 
          Sin[Subscript[\[Theta], 13*"R"]]/E^(I*Subscript[\[Chi]2, "R"])]^2 + 
      Abs[Sin[Subscript[\[Theta], 13*"L"]]/E^(I*Subscript[\[Chi]2, "L"]) - 
          (m["\[Tau]"]*Sin[Subscript[\[Theta], 13*"R"]])/
           (2*E^(I*Subscript[\[Chi]2, "R"])*mbar["B0"]*RV["B0"])]^2*
       Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
           Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] + Cos[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2)/2
 
\[Beta]2["B0", 3, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 0.
 
\[Beta]2["B0", 3, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[uL[1, 1]]^2*Abs[-(m["\[Tau]"]*uL[3, 3])/(2*mbar["B0"]*RV["B0"]) + 
          uR[3, 3]]^2 + Abs[uR[1, 1]]^2*
       Abs[uL[3, 3] - (m["\[Tau]"]*uR[3, 3])/(2*mbar["B0"]*RV["B0"])]^2)/2
 
\[Beta]2["B0", 3, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[(Cos[Subscript[\[Theta], 13*"R"]]*Cos[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
               "R"])) - (Cos[Subscript[\[Theta], 13*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
                "L"]))*mbar["B0"]*RV["B0"])]^2*
       Cos[Subscript[\[Theta], 12*"L"]]^2*Cos[Subscript[\[Theta], 13*"L"]]^
        2 + Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 
              23*"L"]])/E^(I*(Subscript[\[CurlyEpsilon], "L"] - 
              Subscript[\[Chi]1, "L"])) - (Cos[Subscript[\[Theta], 13*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
                "R"]))*mbar["B0"]*RV["B0"])]^2*
       Cos[Subscript[\[Theta], 12*"R"]]^2*Cos[Subscript[\[Theta], 13*"R"]]^2)/
     2
 
\[Beta]2["B0", 3, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 0.
 
\[Beta]2["B0", 3, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[uL[1, 2]]^2*Abs[-(m["\[Tau]"]*uL[3, 3])/(2*mbar["B0"]*RV["B0"]) + 
          uR[3, 3]]^2 + Abs[uR[1, 2]]^2*
       Abs[uL[3, 3] - (m["\[Tau]"]*uR[3, 3])/(2*mbar["B0"]*RV["B0"])]^2)/2
 
\[Beta]2["B0", 3, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[(Cos[Subscript[\[Theta], 13*"R"]]*Cos[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
               "R"])) - (Cos[Subscript[\[Theta], 13*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
                "L"]))*mbar["B0"]*RV["B0"])]^2*
       Cos[Subscript[\[Theta], 13*"L"]]^2*Sin[Subscript[\[Theta], 12*"L"]]^
        2 + Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 
              23*"L"]])/E^(I*(Subscript[\[CurlyEpsilon], "L"] - 
              Subscript[\[Chi]1, "L"])) - (Cos[Subscript[\[Theta], 13*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
                "R"]))*mbar["B0"]*RV["B0"])]^2*
       Cos[Subscript[\[Theta], 13*"R"]]^2*Sin[Subscript[\[Theta], 12*"R"]]^2)/
     2
 
\[Beta]2["B0", 3, 3, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 0.
 
\[Beta]2["B0", 3, 3, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uL[1, 3]] - (Conjugate[uR[1, 3]]*m["\[Tau]"])/
           (2*mbar["B0"]*RV["B0"])]^2*
       Abs[-(m["\[Tau]"]*uL[3, 3])/(2*mbar["B0"]*RV["B0"]) + uR[3, 3]]^2 + 
      Abs[Conjugate[uR[1, 3]] - (Conjugate[uL[1, 3]]*m["\[Tau]"])/
           (2*mbar["B0"]*RV["B0"])]^2*
       Abs[uL[3, 3] - (m["\[Tau]"]*uR[3, 3])/(2*mbar["B0"]*RV["B0"])]^2)/2
 
\[Beta]2["B0", 3, 3, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[(((Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 23*"L"]]*
             m["\[Tau]"])/E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[
                \[Chi]1, "L"])) - (2*Cos[Subscript[\[Theta], 13*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*mbar["B0"]*RV["B0"])/
            E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
                "R"])))^2*((-2*mbar["B0"]*RV["B0"]*Sin[Subscript[\[Theta], 13*
                "L"]])/E^(I*Subscript[\[Chi]2, "L"]) + 
           (m["\[Tau]"]*Sin[Subscript[\[Theta], 13*"R"]])/
            E^(I*Subscript[\[Chi]2, "R"]))^2)/RV["B0"]^4] + 
      Abs[(((Cos[Subscript[\[Theta], 13*"R"]]*Cos[Subscript[\[Theta], 23*
                "R"]]*m["\[Tau]"])/E^(I*(Subscript[\[CurlyEpsilon], "R"] - 
               Subscript[\[Chi]1, "R"])) - 
           (2*Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 23*
                "L"]]*mbar["B0"]*RV["B0"])/E^(I*(Subscript[\[CurlyEpsilon], 
                "L"] - Subscript[\[Chi]1, "L"])))^2*
         ((m["\[Tau]"]*Sin[Subscript[\[Theta], 13*"L"]])/
            E^(I*Subscript[\[Chi]2, "L"]) - (2*mbar["B0"]*RV["B0"]*
             Sin[Subscript[\[Theta], 13*"R"]])/E^(I*Subscript[\[Chi]2, "R"]))^
          2)/RV["B0"]^4])/(32*Abs[mbar["B0"]]^4)
 
\[Beta]2["Bs", 1, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.014049278093934409*Abs[m["e"]]^2*
      Abs[(1.*E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
               "R"]))*m["e"] - 1.3789968659034937*
           E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
               "L"]))*mbar["Bs"]*RV["Bs"])^2/(mbar["Bs"]^4*RV["Bs"]^4)] + 
     (0.026716561804279358*
       Abs[(1.*E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                "L"]))*m["e"] - 2.9006592392646753*
            E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                "R"]))*mbar["Bs"]*RV["Bs"])^2/RV["Bs"]^2])/Abs[mbar["Bs"]]^2
 
\[Beta]2["Bs", 1, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 1]]^2*Abs[uR[2, 1]]^2 + 
      Abs[uL[2, 1]]^2*Abs[uR[3, 1]]^2)/2
 
\[Beta]2["Bs", 1, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*"L"]] + 
          (Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/E^(I*Subscript[\[Delta], "L"])]^
        2*Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] - Sin[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2 + 
      Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] - Sin[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             12*"R"]] + (Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 
              23*"R"]])/E^(I*Subscript[\[Delta], "R"])]^2)/2
 
\[Beta]2["Bs", 1, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (0.02948055057145827*Abs[m["e"]]^2 + 0.02948055057145827*
       Abs[m["\[Mu]"]]^2)*Abs[1/(mbar["Bs"]^2*RV["Bs"]^2)]
 
\[Beta]2["Bs", 1, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 1]]^2*Abs[uR[2, 2]]^2 + 
      Abs[uL[2, 2]]^2*Abs[uR[3, 1]]^2)/2
 
\[Beta]2["Bs", 1, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 23*"L"]] - 
          (Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/E^(I*Subscript[\[Delta], "L"])]^
        2*Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] - Sin[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2 + 
      Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] - Sin[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 
             23*"R"]] - (Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 
              23*"R"]])/E^(I*Subscript[\[Delta], "R"])]^2)/2
 
\[Beta]2["Bs", 1, 3, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.002344219123474507*Abs[m["e"]]^2*
      Abs[(1.*m["\[Tau]"] - 2.*mbar["Bs"]*RV["Bs"])^2/
        (mbar["Bs"]^4*RV["Bs"]^4)] + (0.009376876493898028*
       Abs[(1.*m["\[Tau]"] - 2.*mbar["Bs"]*RV["Bs"])^2/RV["Bs"]^2])/
      Abs[mbar["Bs"]]^2
 
\[Beta]2["Bs", 1, 3, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[2, 3]] - (Conjugate[uL[2, 3]]*m["\[Tau]"])/
           (2*mbar["Bs"]*RV["Bs"])]^2*Abs[uL[3, 1]]^2 + 
      Abs[Conjugate[uL[2, 3]] - (Conjugate[uR[2, 3]]*m["\[Tau]"])/
           (2*mbar["Bs"]*RV["Bs"])]^2*Abs[uR[3, 1]]^2)/2
 
\[Beta]2["Bs", 1, 3, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] - Sin[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[-(Cos[Subscript[\[Theta], 13*"L"]]*m["\[Tau]"]*
             Sin[Subscript[\[Theta], 23*"L"]])/
           (2*E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], 
                "L"] + Subscript[\[Chi]2, "L"]))*mbar["Bs"]*RV["Bs"]) + 
          (Cos[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
              Subscript[\[Chi]2, "R"]))]^2 + 
      Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 23*"L"]])/
           E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
              Subscript[\[Chi]2, "L"])) - (Cos[Subscript[\[Theta], 13*"R"]]*
            m["\[Tau]"]*Sin[Subscript[\[Theta], 23*"R"]])/
           (2*E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], 
                "R"] + Subscript[\[Chi]2, "R"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] - Sin[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2)/2
 
\[Beta]2["Bs", 2, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (0.10712538871064461*
       Abs[((1.*E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*m["e"] - 1.3789968659034937*
             E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*mbar["Bs"]*RV["Bs"])^2*
          (0.3447492164758735*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + 
                Subscript[\[CurlyPhi]1, "L"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*mbar["Bs"]*RV["Bs"])^2)/RV["Bs"]^4] + 
      0.024211714530627627*
       Abs[((0.7251648098161688*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + 
                Subscript[\[CurlyPhi]1, "R"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*mbar["Bs"]*RV["Bs"])^2*
          (1.*E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*m["e"] - 2.9006592392646753*
             E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*mbar["Bs"]*RV["Bs"])^2)/RV["Bs"]^4])/Abs[mbar["Bs"]]^4
 
\[Beta]2["Bs", 2, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 2]]^2*Abs[uR[2, 1]]^2 + 
      Abs[uL[2, 1]]^2*Abs[uR[3, 2]]^2)/2
 
\[Beta]2["Bs", 2, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*"L"]] + 
          (Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/E^(I*Subscript[\[Delta], "L"])]^
        2*Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
           Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] + Cos[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2 + 
      Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
           Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] + Cos[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             12*"R"]] + (Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 
              23*"R"]])/E^(I*Subscript[\[Delta], "R"])]^2)/2
 
\[Beta]2["Bs", 2, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.014049278093934406*Abs[m["\[Mu]"]]^2*
      Abs[(1.*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
               "R"]))*m["\[Mu]"] - 1.378996865903494*
           E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
               "L"]))*mbar["Bs"]*RV["Bs"])^2/(mbar["Bs"]^4*RV["Bs"]^4)] + 
     (0.026716561804279358*
       Abs[(1.*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                "L"]))*m["\[Mu]"] - 2.9006592392646753*
            E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                "R"]))*mbar["Bs"]*RV["Bs"])^2/RV["Bs"]^2])/Abs[mbar["Bs"]]^2
 
\[Beta]2["Bs", 2, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 2]]^2*Abs[uR[2, 2]]^2 + 
      Abs[uL[2, 2]]^2*Abs[uR[3, 2]]^2)/2
 
\[Beta]2["Bs", 2, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 23*"L"]] - 
          (Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/E^(I*Subscript[\[Delta], "L"])]^
        2*Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
           Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] + Cos[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2 + 
      Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
           Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] + Cos[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 
             23*"R"]] - (Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 
              23*"R"]])/E^(I*Subscript[\[Delta], "R"])]^2)/2
 
\[Beta]2["Bs", 2, 3, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (0.008497734795457351*Abs[((1.*m["\[Tau]"] - 2.*mbar["Bs"]*RV["Bs"])^2*
          (0.7251648098161688*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + 
                Subscript[\[CurlyPhi]1, "R"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*mbar["Bs"]*RV["Bs"])^2)/RV["Bs"]^4] + 
      0.01787461128935538*Abs[((1.*m["\[Tau]"] - 2.*mbar["Bs"]*RV["Bs"])^2*
          (0.3447492164758735*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + 
                Subscript[\[CurlyPhi]1, "L"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*mbar["Bs"]*RV["Bs"])^2)/RV["Bs"]^4])/Abs[mbar["Bs"]]^4
 
\[Beta]2["Bs", 2, 3, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[2, 3]] - (Conjugate[uL[2, 3]]*m["\[Tau]"])/
           (2*mbar["Bs"]*RV["Bs"])]^2*Abs[uL[3, 2]]^2 + 
      Abs[Conjugate[uL[2, 3]] - (Conjugate[uR[2, 3]]*m["\[Tau]"])/
           (2*mbar["Bs"]*RV["Bs"])]^2*Abs[uR[3, 2]]^2)/2
 
\[Beta]2["Bs", 2, 3, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
           Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] + Cos[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[-(Cos[Subscript[\[Theta], 13*"L"]]*m["\[Tau]"]*
             Sin[Subscript[\[Theta], 23*"L"]])/
           (2*E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], 
                "L"] + Subscript[\[Chi]2, "L"]))*mbar["Bs"]*RV["Bs"]) + 
          (Cos[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
              Subscript[\[Chi]2, "R"]))]^2 + 
      Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
           Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] + Cos[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2*
       Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 
              23*"L"]])/E^(I*(Subscript[\[Delta], "L"] + Subscript[
               \[CurlyEpsilon], "L"] + Subscript[\[Chi]2, "L"])) - 
          (Cos[Subscript[\[Theta], 13*"R"]]*m["\[Tau]"]*
            Sin[Subscript[\[Theta], 23*"R"]])/
           (2*E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], 
                "R"] + Subscript[\[Chi]2, "R"]))*mbar["Bs"]*RV["Bs"])]^2)/2
 
\[Beta]2["Bs", 3, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 0.
 
\[Beta]2["Bs", 3, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[uL[2, 1]]^2*Abs[-(m["\[Tau]"]*uL[3, 3])/(2*mbar["Bs"]*RV["Bs"]) + 
          uR[3, 3]]^2 + Abs[uR[2, 1]]^2*
       Abs[uL[3, 3] - (m["\[Tau]"]*uR[3, 3])/(2*mbar["Bs"]*RV["Bs"])]^2)/2
 
\[Beta]2["Bs", 3, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[(Cos[Subscript[\[Theta], 13*"R"]]*Cos[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
               "R"])) - (Cos[Subscript[\[Theta], 13*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
                "L"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
             12*"L"]] + (Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 
              23*"L"]])/E^(I*Subscript[\[Delta], "L"])]^2 + 
      Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 23*"L"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
               "L"])) - (Cos[Subscript[\[Theta], 13*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
                "R"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             12*"R"]] + (Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 
              23*"R"]])/E^(I*Subscript[\[Delta], "R"])]^2)/2
 
\[Beta]2["Bs", 3, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 0.
 
\[Beta]2["Bs", 3, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[uL[2, 2]]^2*Abs[-(m["\[Tau]"]*uL[3, 3])/(2*mbar["Bs"]*RV["Bs"]) + 
          uR[3, 3]]^2 + Abs[uR[2, 2]]^2*
       Abs[uL[3, 3] - (m["\[Tau]"]*uR[3, 3])/(2*mbar["Bs"]*RV["Bs"])]^2)/2
 
\[Beta]2["Bs", 3, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[(Cos[Subscript[\[Theta], 13*"R"]]*Cos[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
               "R"])) - (Cos[Subscript[\[Theta], 13*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
                "L"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 
             23*"L"]] - (Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 
              23*"L"]])/E^(I*Subscript[\[Delta], "L"])]^2 + 
      Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 23*"L"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
               "L"])) - (Cos[Subscript[\[Theta], 13*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
                "R"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 
             23*"R"]] - (Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 
              23*"R"]])/E^(I*Subscript[\[Delta], "R"])]^2)/2
 
\[Beta]2["Bs", 3, 3, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 0.
 
\[Beta]2["Bs", 3, 3, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uL[2, 3]] - (Conjugate[uR[2, 3]]*m["\[Tau]"])/
           (2*mbar["Bs"]*RV["Bs"])]^2*
       Abs[-(m["\[Tau]"]*uL[3, 3])/(2*mbar["Bs"]*RV["Bs"]) + uR[3, 3]]^2 + 
      Abs[Conjugate[uR[2, 3]] - (Conjugate[uL[2, 3]]*m["\[Tau]"])/
           (2*mbar["Bs"]*RV["Bs"])]^2*
       Abs[uL[3, 3] - (m["\[Tau]"]*uR[3, 3])/(2*mbar["Bs"]*RV["Bs"])]^2)/2
 
\[Beta]2["Bs", 3, 3, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 23*"L"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
               "L"])) - (Cos[Subscript[\[Theta], 13*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
                "R"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[-(Cos[Subscript[\[Theta], 13*"L"]]*m["\[Tau]"]*
             Sin[Subscript[\[Theta], 23*"L"]])/
           (2*E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], 
                "L"] + Subscript[\[Chi]2, "L"]))*mbar["Bs"]*RV["Bs"]) + 
          (Cos[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
              Subscript[\[Chi]2, "R"]))]^2 + 
      Abs[(Cos[Subscript[\[Theta], 13*"R"]]*Cos[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
               "R"])) - (Cos[Subscript[\[Theta], 13*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
                "L"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 
              23*"L"]])/E^(I*(Subscript[\[Delta], "L"] + Subscript[
               \[CurlyEpsilon], "L"] + Subscript[\[Chi]2, "L"])) - 
          (Cos[Subscript[\[Theta], 13*"R"]]*m["\[Tau]"]*
            Sin[Subscript[\[Theta], 23*"R"]])/
           (2*E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], 
                "R"] + Subscript[\[Chi]2, "R"]))*mbar["Bs"]*RV["Bs"])]^2)/2
 
\[Beta]2["K0L", 1, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (Abs[(0. + 8.326672684688674*^-17*I)/E^(I*(Subscript[\[CurlyPhi]0, "L"] - 
             Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[CurlyPhi]1, "R"])) + 
         (m["e"]*(mbar["K0"]*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["K0"]*RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             ((0. - 0.25824781755683757*I)*mbar["K0"]*RV["K0"] + 
              (0. + 0.25824781755683757*I)*mbar[
                "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*RV[
                "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]) + 
            (0. + 0.06034312365542037*I)*E^(I*(Subscript[\[CurlyPhi]0, "L"] - 
                Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "L"] - 
                Subscript[\[CurlyPhi]1, "R"]))*m["e"]*
             (1.*mbar["K0"]^2*RV["K0"]^2 - 1.0000000000000002*
               mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
               RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)))/
          (mbar["K0"]^2*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
           RV["K0"]^2*RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)]^2 + 
      Abs[(0. - 8.326672684688674*^-17*I)*
          E^(I*(Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "L"] - Subscript[\[CurlyPhi]1, "R"])) + 
         (m["e"]*(mbar["K0"]*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["K0"]*RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             ((0. - 0.25824781755683757*I)*mbar["K0"]*RV["K0"] + 
              (0. + 0.25824781755683757*I)*mbar[
                "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*RV[
                "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]) + 
            ((0. + 0.06034312365542038*I)*m["e"]*(1.*mbar["K0"]^2*
                RV["K0"]^2 - 0.9999999999999998*
                mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
                RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2))/
             E^(I*(Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]0, 
                 "R"] + Subscript[\[CurlyPhi]1, "L"] - Subscript[
                 \[CurlyPhi]1, "R"]))))/(mbar["K0"]^2*
           mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*RV["K0"]^2*
           RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)]^2)/4
 
\[Beta]2["K0L", 1, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[2, 1]]*uL[1, 1] + Conjugate[uR[1, 1]]*uL[2, 1]]^2 + 
      Abs[Conjugate[uL[2, 1]]*uR[1, 1] + Conjugate[uL[1, 1]]*uR[2, 1]]^2)/4
 
\[Beta]2["K0L", 1, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 12*"R"]]*
          Cos[Subscript[\[Theta], 13*"R"]]*(Cos[Subscript[\[Theta], 23*"L"]]*
            Sin[Subscript[\[Theta], 12*"L"]] + E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]) + 
         Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]]*
          (Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
              12*"R"]] + (Cos[Subscript[\[Theta], 12*"R"]]*
             Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 23*
                "R"]])/E^(I*Subscript[\[Delta], "R"]))]^2 + 
      Abs[Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]]*
          (Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
              12*"L"]] + (Cos[Subscript[\[Theta], 12*"L"]]*
             Sin[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 23*
                "L"]])/E^(I*Subscript[\[Delta], "L"])) + 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 12*"L"]]*
          Cos[Subscript[\[Theta], 13*"L"]]*(Cos[Subscript[\[Theta], 23*"R"]]*
            Sin[Subscript[\[Theta], 12*"R"]] + E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])]^2)/4
 
\[Beta]2["K0L", 1, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.25*(Abs[(0. + 5.551115123125783*^-17*I)*
          E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
              "L"])) + ((0. + 0.08741150105010506*I)*
           E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
               "R"]))*m["e"]*(1.*mbar["K0"]*RV["K0"] - 
            1.*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]))/
          (mbar["K0"]*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*RV["K0"]*
           RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2 + 
      0.004018006157202129*Abs[m["\[Mu]"]]^2*
       Abs[(E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                "R"]))*mbar["K0"]*mbar[
             "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*RV["K0"]*
            RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
            (-1.3789968659034941*mbar["K0"]*RV["K0"] + 1.3789968659034941*
              mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
              RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]) + 
           E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                "L"]))*m["e"]*(1.*mbar["K0"]^2*RV["K0"]^2 - 
             1.*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
              RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2))/
          (mbar["K0"]^2*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
           RV["K0"]^2*RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)]^2)
 
\[Beta]2["K0L", 1, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[2, 2]]*uL[1, 1] + Conjugate[uR[1, 2]]*uL[2, 1]]^2 + 
      Abs[Conjugate[uL[2, 2]]*uR[1, 1] + Conjugate[uL[1, 2]]*uR[2, 1]]^2)/4
 
\[Beta]2["K0L", 1, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]]*
          (Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 
              23*"L"]] - (Sin[Subscript[\[Theta], 12*"L"]]*
             Sin[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 23*
                "L"]])/E^(I*Subscript[\[Delta], "L"])) - 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 12*"L"]]*(Cos[Subscript[\[Theta], 23*"R"]]*
            Sin[Subscript[\[Theta], 12*"R"]] + E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])]^2 + 
      Abs[E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          (Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
              12*"L"]] + E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])*Sin[Subscript[\[Theta], 
            12*"R"]] - Cos[Subscript[\[Theta], 12*"L"]]*
          Cos[Subscript[\[Theta], 13*"L"]]*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - 
           (Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
             Sin[Subscript[\[Theta], 23*"R"]])/
            E^(I*Subscript[\[Delta], "R"]))]^2)/4
 
\[Beta]2["K0L", 2, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.0019101926289581295*Abs[m["e"]]^2*
      Abs[1./(mbar["K0"]*RV["K0"]) - 
         1./(mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
           RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2 + 
     0.0010045015393005327*Abs[m["\[Mu]"]]^2*
      Abs[(E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
               "L"]))*mbar["K0"]*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
           RV["K0"]*RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
           (-1.3789968659034937*mbar["K0"]*RV["K0"] + 1.3789968659034937*
             mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]) + 
          E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
           m["e"]*(1.*mbar["K0"]^2*RV["K0"]^2 - 0.9999999999999998*
             mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
             RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2))/
         (mbar["K0"]^2*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
          RV["K0"]^2*RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)]^2
 
\[Beta]2["K0L", 2, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[2, 1]]*uL[1, 2] + Conjugate[uR[1, 1]]*uL[2, 2]]^2 + 
      Abs[Conjugate[uL[2, 1]]*uR[1, 2] + Conjugate[uL[1, 1]]*uR[2, 2]]^2)/4
 
\[Beta]2["K0L", 2, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 12*"R"]]*
          Cos[Subscript[\[Theta], 13*"R"]]*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]) - 
         Cos[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]]*
          (Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
              12*"R"]] + (Cos[Subscript[\[Theta], 12*"R"]]*
             Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 23*
                "R"]])/E^(I*Subscript[\[Delta], "R"]))]^2 + 
      Abs[Cos[Subscript[\[Theta], 13*"R"]]*(Cos[Subscript[\[Theta], 23*"L"]]*
            Sin[Subscript[\[Theta], 12*"L"]] + 
           (Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
             Sin[Subscript[\[Theta], 23*"L"]])/
            E^(I*Subscript[\[Delta], "L"]))*Sin[Subscript[\[Theta], 
            12*"R"]] - E^(I*(Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[CurlyEpsilon], "R"]))*Cos[Subscript[\[Theta], 
            12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]]*
          (Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 
              23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])]^2)/4
 
\[Beta]2["K0L", 2, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (Abs[m["\[Mu]"]]^2*Abs[(0. + 0.09182186075753421*I)/
          (mbar["K0"]*RV["K0"]) - (0. + 0.09182186075753421*I)/
          (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
           RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2)/2
 
\[Beta]2["K0L", 2, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          (Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 
              23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])*Sin[Subscript[\[Theta], 
            12*"R"]] + Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 12*"L"]]*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - 
           (Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
             Sin[Subscript[\[Theta], 23*"R"]])/
            E^(I*Subscript[\[Delta], "R"]))]^2 + 
      Abs[Cos[Subscript[\[Theta], 13*"R"]]*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - 
           (Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
             Sin[Subscript[\[Theta], 23*"L"]])/
            E^(I*Subscript[\[Delta], "L"]))*Sin[Subscript[\[Theta], 
            12*"R"]] + E^(I*(Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[CurlyEpsilon], "R"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]]*
          (Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 
              23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])]^2)/4
 
\[Beta]2["B0", 1, 1, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    0.2622959554785427*Abs[0.3781492963299588 - (0.1303666736206562*m["e"])/
          (mbar["B0"]*RV["B0"])]^2 + 0.002344219123474507*Abs[m["e"]]^2*
      Abs[(1.*m["e"] - 1.3789968659034941*mbar["B0"]*RV["B0"])^2/
        (mbar["B0"]^4*RV["B0"]^4)]
 
\[Beta]2["B0", 1, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.002344219123474507*Abs[m["e"]]^2*
      Abs[(1.*E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
               "R"]))*m["e"] - 1.3789968659034941*
           E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
               "L"]))*mbar["B0"]*RV["B0"])^2/(mbar["B0"]^4*RV["B0"]^4)] + 
     (0.004457842935155488*
       Abs[(1.*E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                "L"]))*m["e"] - 2.900659239264675*
            E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                "R"]))*mbar["B0"]*RV["B0"])^2/RV["B0"]^2])/Abs[mbar["B0"]]^2
 
\[Beta]2["B0", 1, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 1]]^2*Abs[uR[1, 1]]^2 + 
      Abs[uL[1, 1]]^2*Abs[uR[3, 1]]^2)/2
 
\[Beta]2["B0", 1, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] - Sin[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2*
       Cos[Subscript[\[Theta], 12*"L"]]^2*Cos[Subscript[\[Theta], 13*"L"]]^
        2 + Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 
             12*"L"]]*Cos[Subscript[\[Theta], 23*"L"]]*
           Sin[Subscript[\[Theta], 13*"L"]] - 
          Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 23*"L"]]]^
        2*Cos[Subscript[\[Theta], 12*"R"]]^2*Cos[Subscript[\[Theta], 13*"R"]]^
        2)/2
 
\[Beta]2["B0", 1, 2, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (0.004919033558742541*Abs[m["e"]]^2 + 0.004919033558742541*
       Abs[m["\[Mu]"]]^2)*Abs[1/(mbar["B0"]*RV["B0"])]^2
 
\[Beta]2["B0", 1, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (0.004919033558742541*Abs[m["e"]]^2 + 0.004919033558742541*
       Abs[m["\[Mu]"]]^2)*Abs[1/(mbar["B0"]^2*RV["B0"]^2)]
 
\[Beta]2["B0", 1, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 1]]^2*Abs[uR[1, 2]]^2 + 
      Abs[uL[1, 2]]^2*Abs[uR[3, 1]]^2)/2
 
\[Beta]2["B0", 1, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] - Sin[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2*
       Cos[Subscript[\[Theta], 13*"L"]]^2*Sin[Subscript[\[Theta], 12*"L"]]^
        2 + Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 
             12*"L"]]*Cos[Subscript[\[Theta], 23*"L"]]*
           Sin[Subscript[\[Theta], 13*"L"]] - 
          Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 23*"L"]]]^
        2*Cos[Subscript[\[Theta], 13*"R"]]^2*Sin[Subscript[\[Theta], 12*"R"]]^
        2)/2
 
\[Beta]2["B0", 1, 3, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    0.2622959554785427*Abs[0.9257446244430247 - 
         (0.46287231222151237*m["\[Tau]"])/(mbar["B0"]*RV["B0"])]^2 + 
     0.056197112375737636*Abs[m["e"]]^2*
      Abs[(0.5*m["\[Tau]"] - 1.*mbar["B0"]*RV["B0"])^2/
        (mbar["B0"]^4*RV["B0"]^4)]
 
\[Beta]2["B0", 1, 3, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.014049278093934409*Abs[m["e"]]^2*
      Abs[(1.*m["\[Tau]"] - 2.*mbar["B0"]*RV["B0"])^2/
        (mbar["B0"]^4*RV["B0"]^4)] + (0.056197112375737636*
       Abs[(1.*m["\[Tau]"] - 2.*mbar["B0"]*RV["B0"])^2/RV["B0"]^2])/
      Abs[mbar["B0"]]^2
 
\[Beta]2["B0", 1, 3, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[1, 3]] - (Conjugate[uL[1, 3]]*m["\[Tau]"])/
           (2*mbar["B0"]*RV["B0"])]^2*Abs[uL[3, 1]]^2 + 
      Abs[Conjugate[uL[1, 3]] - (Conjugate[uR[1, 3]]*m["\[Tau]"])/
           (2*mbar["B0"]*RV["B0"])]^2*Abs[uR[3, 1]]^2)/2
 
\[Beta]2["B0", 1, 3, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] - Sin[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[-(m["\[Tau]"]*Sin[Subscript[\[Theta], 13*"L"]])/
           (2*E^(I*Subscript[\[Chi]2, "L"])*mbar["B0"]*RV["B0"]) + 
          Sin[Subscript[\[Theta], 13*"R"]]/E^(I*Subscript[\[Chi]2, "R"])]^2 + 
      Abs[Sin[Subscript[\[Theta], 13*"L"]]/E^(I*Subscript[\[Chi]2, "L"]) - 
          (m["\[Tau]"]*Sin[Subscript[\[Theta], 13*"R"]])/
           (2*E^(I*Subscript[\[Chi]2, "R"])*mbar["B0"]*RV["B0"])]^2*
       Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] - Sin[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2)/2
 
\[Beta]2["B0", 2, 1, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (0.033990939181829403*Abs[((0.7251648098161687*m["e"] - 
            1.*mbar["B0"]*RV["B0"])^2*(0.3447492164758735*m["\[Mu]"] - 
            1.*mbar["B0"]*RV["B0"])^2)/RV["B0"]^4] + 
      0.03399093918182939*Abs[((0.34474921647587353*m["e"] - 
            1.*mbar["B0"]*RV["B0"])^2*(0.7251648098161688*m["\[Mu]"] - 
            1.*mbar["B0"]*RV["B0"])^2)/RV["B0"]^4])/Abs[mbar["B0"]]^4
 
\[Beta]2["B0", 2, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (0.017874611289355375*
       Abs[((1.*E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*m["e"] - 1.3789968659034941*
             E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*mbar["B0"]*RV["B0"])^2*
          (0.3447492164758735*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + 
                Subscript[\[CurlyPhi]1, "L"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*mbar["B0"]*RV["B0"])^2)/RV["B0"]^4] + 
      0.004039891860301862*
       Abs[((0.7251648098161688*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + 
                Subscript[\[CurlyPhi]1, "R"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*mbar["B0"]*RV["B0"])^2*
          (1.*E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*m["e"] - 2.900659239264675*
             E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*mbar["B0"]*RV["B0"])^2)/RV["B0"]^4])/Abs[mbar["B0"]]^4
 
\[Beta]2["B0", 2, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 2]]^2*Abs[uR[1, 1]]^2 + 
      Abs[uL[1, 1]]^2*Abs[uR[3, 2]]^2)/2
 
\[Beta]2["B0", 2, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
           Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] + Cos[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2*
       Cos[Subscript[\[Theta], 12*"L"]]^2*Cos[Subscript[\[Theta], 13*"L"]]^
        2 + Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 
             23*"L"]]*Sin[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 13*"L"]] + 
          Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 23*"L"]]]^
        2*Cos[Subscript[\[Theta], 12*"R"]]^2*Cos[Subscript[\[Theta], 13*"R"]]^
        2)/2
 
\[Beta]2["B0", 2, 2, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    0.03750750597559211*Abs[(0. + 1.*I) - ((0. + 0.3447492164758735*I)*
           m["\[Mu]"])/(mbar["B0"]*RV["B0"])]^2 + 0.002344219123474507*
      Abs[m["\[Mu]"]]^2*Abs[(1.*m["\[Mu]"] - 1.378996865903494*mbar["B0"]*
           RV["B0"])^2/(mbar["B0"]^4*RV["B0"]^4)]
 
\[Beta]2["B0", 2, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.002344219123474507*Abs[m["\[Mu]"]]^2*
      Abs[(1.*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
               "R"]))*m["\[Mu]"] - 1.378996865903494*
           E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
               "L"]))*mbar["B0"]*RV["B0"])^2/(mbar["B0"]^4*RV["B0"]^4)] + 
     (0.004457842935155487*
       Abs[(1.*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                "L"]))*m["\[Mu]"] - 2.9006592392646753*
            E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                "R"]))*mbar["B0"]*RV["B0"])^2/RV["B0"]^2])/Abs[mbar["B0"]]^2
 
\[Beta]2["B0", 2, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 2]]^2*Abs[uR[1, 2]]^2 + 
      Abs[uL[1, 2]]^2*Abs[uR[3, 2]]^2)/2
 
\[Beta]2["B0", 2, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
           Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] + Cos[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2*
       Cos[Subscript[\[Theta], 13*"L"]]^2*Sin[Subscript[\[Theta], 12*"L"]]^
        2 + Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 
             23*"L"]]*Sin[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 13*"L"]] + 
          Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 23*"L"]]]^
        2*Cos[Subscript[\[Theta], 13*"R"]]^2*Sin[Subscript[\[Theta], 12*"R"]]^
        2)/2
 
\[Beta]2["B0", 2, 3, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (0.4285015548425785*Abs[((0.3447492164758735*m["\[Mu]"] - 
            1.*mbar["B0"]*RV["B0"])^2*(0.5*m["\[Tau]"] - 1.*mbar["B0"]*
             RV["B0"])^2)/RV["B0"]^4] + 0.20371310533962794*
       Abs[((0.7251648098161688*m["\[Mu]"] - 1.*mbar["B0"]*RV["B0"])^2*
          (0.5*m["\[Tau]"] - 1.*mbar["B0"]*RV["B0"])^2)/RV["B0"]^4])/
     Abs[mbar["B0"]]^4
 
\[Beta]2["B0", 2, 3, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (0.050928276334906986*Abs[((1.*m["\[Tau]"] - 2.*mbar["B0"]*RV["B0"])^2*
          (0.7251648098161688*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + 
                Subscript[\[CurlyPhi]1, "R"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*mbar["B0"]*RV["B0"])^2)/RV["B0"]^4] + 
      0.10712538871064463*Abs[((1.*m["\[Tau]"] - 2.*mbar["B0"]*RV["B0"])^2*
          (0.3447492164758735*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + 
                Subscript[\[CurlyPhi]1, "L"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*mbar["B0"]*RV["B0"])^2)/RV["B0"]^4])/Abs[mbar["B0"]]^4
 
\[Beta]2["B0", 2, 3, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[1, 3]] - (Conjugate[uL[1, 3]]*m["\[Tau]"])/
           (2*mbar["B0"]*RV["B0"])]^2*Abs[uL[3, 2]]^2 + 
      Abs[Conjugate[uL[1, 3]] - (Conjugate[uR[1, 3]]*m["\[Tau]"])/
           (2*mbar["B0"]*RV["B0"])]^2*Abs[uR[3, 2]]^2)/2
 
\[Beta]2["B0", 2, 3, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
           Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] + Cos[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[-(m["\[Tau]"]*Sin[Subscript[\[Theta], 13*"L"]])/
           (2*E^(I*Subscript[\[Chi]2, "L"])*mbar["B0"]*RV["B0"]) + 
          Sin[Subscript[\[Theta], 13*"R"]]/E^(I*Subscript[\[Chi]2, "R"])]^2 + 
      Abs[Sin[Subscript[\[Theta], 13*"L"]]/E^(I*Subscript[\[Chi]2, "L"]) - 
          (m["\[Tau]"]*Sin[Subscript[\[Theta], 13*"R"]])/
           (2*E^(I*Subscript[\[Chi]2, "R"])*mbar["B0"]*RV["B0"])]^2*
       Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
           Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] + Cos[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2)/2
 
\[Beta]2["B0", 3, 1, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 0.
 
\[Beta]2["B0", 3, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 0.
 
\[Beta]2["B0", 3, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[uL[1, 1]]^2*Abs[-(m["\[Tau]"]*uL[3, 3])/(2*mbar["B0"]*RV["B0"]) + 
          uR[3, 3]]^2 + Abs[uR[1, 1]]^2*
       Abs[uL[3, 3] - (m["\[Tau]"]*uR[3, 3])/(2*mbar["B0"]*RV["B0"])]^2)/2
 
\[Beta]2["B0", 3, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[(Cos[Subscript[\[Theta], 13*"R"]]*Cos[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
               "R"])) - (Cos[Subscript[\[Theta], 13*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
                "L"]))*mbar["B0"]*RV["B0"])]^2*
       Cos[Subscript[\[Theta], 12*"L"]]^2*Cos[Subscript[\[Theta], 13*"L"]]^
        2 + Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 
              23*"L"]])/E^(I*(Subscript[\[CurlyEpsilon], "L"] - 
              Subscript[\[Chi]1, "L"])) - (Cos[Subscript[\[Theta], 13*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
                "R"]))*mbar["B0"]*RV["B0"])]^2*
       Cos[Subscript[\[Theta], 12*"R"]]^2*Cos[Subscript[\[Theta], 13*"R"]]^2)/
     2
 
\[Beta]2["B0", 3, 2, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 0.
 
\[Beta]2["B0", 3, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 0.
 
\[Beta]2["B0", 3, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[uL[1, 2]]^2*Abs[-(m["\[Tau]"]*uL[3, 3])/(2*mbar["B0"]*RV["B0"]) + 
          uR[3, 3]]^2 + Abs[uR[1, 2]]^2*
       Abs[uL[3, 3] - (m["\[Tau]"]*uR[3, 3])/(2*mbar["B0"]*RV["B0"])]^2)/2
 
\[Beta]2["B0", 3, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[(Cos[Subscript[\[Theta], 13*"R"]]*Cos[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
               "R"])) - (Cos[Subscript[\[Theta], 13*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
                "L"]))*mbar["B0"]*RV["B0"])]^2*
       Cos[Subscript[\[Theta], 13*"L"]]^2*Sin[Subscript[\[Theta], 12*"L"]]^
        2 + Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 
              23*"L"]])/E^(I*(Subscript[\[CurlyEpsilon], "L"] - 
              Subscript[\[Chi]1, "L"])) - (Cos[Subscript[\[Theta], 13*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
                "R"]))*mbar["B0"]*RV["B0"])]^2*
       Cos[Subscript[\[Theta], 13*"R"]]^2*Sin[Subscript[\[Theta], 12*"R"]]^2)/
     2
 
\[Beta]2["B0", 3, 3, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 0.
 
\[Beta]2["B0", 3, 3, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 0.
 
\[Beta]2["B0", 3, 3, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uL[1, 3]] - (Conjugate[uR[1, 3]]*m["\[Tau]"])/
           (2*mbar["B0"]*RV["B0"])]^2*
       Abs[-(m["\[Tau]"]*uL[3, 3])/(2*mbar["B0"]*RV["B0"]) + uR[3, 3]]^2 + 
      Abs[Conjugate[uR[1, 3]] - (Conjugate[uL[1, 3]]*m["\[Tau]"])/
           (2*mbar["B0"]*RV["B0"])]^2*
       Abs[uL[3, 3] - (m["\[Tau]"]*uR[3, 3])/(2*mbar["B0"]*RV["B0"])]^2)/2
 
\[Beta]2["B0", 3, 3, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[(((Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 23*"L"]]*
             m["\[Tau]"])/E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[
                \[Chi]1, "L"])) - (2*Cos[Subscript[\[Theta], 13*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*mbar["B0"]*RV["B0"])/
            E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
                "R"])))^2*((-2*mbar["B0"]*RV["B0"]*Sin[Subscript[\[Theta], 13*
                "L"]])/E^(I*Subscript[\[Chi]2, "L"]) + 
           (m["\[Tau]"]*Sin[Subscript[\[Theta], 13*"R"]])/
            E^(I*Subscript[\[Chi]2, "R"]))^2)/RV["B0"]^4] + 
      Abs[(((Cos[Subscript[\[Theta], 13*"R"]]*Cos[Subscript[\[Theta], 23*
                "R"]]*m["\[Tau]"])/E^(I*(Subscript[\[CurlyEpsilon], "R"] - 
               Subscript[\[Chi]1, "R"])) - 
           (2*Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 23*
                "L"]]*mbar["B0"]*RV["B0"])/E^(I*(Subscript[\[CurlyEpsilon], 
                "L"] - Subscript[\[Chi]1, "L"])))^2*
         ((m["\[Tau]"]*Sin[Subscript[\[Theta], 13*"L"]])/
            E^(I*Subscript[\[Chi]2, "L"]) - (2*mbar["B0"]*RV["B0"]*
             Sin[Subscript[\[Theta], 13*"R"]])/E^(I*Subscript[\[Chi]2, "R"]))^
          2)/RV["B0"]^4])/(32*Abs[mbar["B0"]]^4)
 
\[Beta]2["Bs", 1, 1, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    0.2622959554785427*Abs[(0. + 0.9257446244430247*I) - 
         ((0. + 0.3191497339334845*I)*m["e"])/(mbar["Bs"]*RV["Bs"])]^2 + 
     0.014049278093934409*Abs[m["e"]]^2*
      Abs[(1.*m["e"] - 1.3789968659034937*mbar["Bs"]*RV["Bs"])^2/
        (mbar["Bs"]^4*RV["Bs"]^4)]
 
\[Beta]2["Bs", 1, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.014049278093934409*Abs[m["e"]]^2*
      Abs[(1.*E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
               "R"]))*m["e"] - 1.3789968659034937*
           E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
               "L"]))*mbar["Bs"]*RV["Bs"])^2/(mbar["Bs"]^4*RV["Bs"]^4)] + 
     (0.026716561804279358*
       Abs[(1.*E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                "L"]))*m["e"] - 2.9006592392646753*
            E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                "R"]))*mbar["Bs"]*RV["Bs"])^2/RV["Bs"]^2])/Abs[mbar["Bs"]]^2
 
\[Beta]2["Bs", 1, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 1]]^2*Abs[uR[2, 1]]^2 + 
      Abs[uL[2, 1]]^2*Abs[uR[3, 1]]^2)/2
 
\[Beta]2["Bs", 1, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*"L"]] + 
          (Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/E^(I*Subscript[\[Delta], "L"])]^
        2*Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] - Sin[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2 + 
      Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] - Sin[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             12*"R"]] + (Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 
              23*"R"]])/E^(I*Subscript[\[Delta], "R"])]^2)/2
 
\[Beta]2["Bs", 1, 2, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (0.02948055057145827*Abs[m["e"]]^2 + 0.02948055057145827*
       Abs[m["\[Mu]"]]^2)*Abs[1/(mbar["Bs"]*RV["Bs"])]^2
 
\[Beta]2["Bs", 1, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (0.02948055057145827*Abs[m["e"]]^2 + 0.02948055057145827*
       Abs[m["\[Mu]"]]^2)*Abs[1/(mbar["Bs"]^2*RV["Bs"]^2)]
 
\[Beta]2["Bs", 1, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 1]]^2*Abs[uR[2, 2]]^2 + 
      Abs[uL[2, 2]]^2*Abs[uR[3, 1]]^2)/2
 
\[Beta]2["Bs", 1, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 23*"L"]] - 
          (Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/E^(I*Subscript[\[Delta], "L"])]^
        2*Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] - Sin[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2 + 
      Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] - Sin[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 
             23*"R"]] - (Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 
              23*"R"]])/E^(I*Subscript[\[Delta], "R"])]^2)/2
 
\[Beta]2["Bs", 1, 3, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    0.2622959554785427*Abs[(0. + 0.3781492963299588*I) - 
         ((0. + 0.1890746481649794*I)*m["\[Tau]"])/(mbar["Bs"]*RV["Bs"])]^2 + 
     0.009376876493898028*Abs[m["e"]]^2*
      Abs[(0.5*m["\[Tau]"] - 1.*mbar["Bs"]*RV["Bs"])^2/
        (mbar["Bs"]^4*RV["Bs"]^4)]
 
\[Beta]2["Bs", 1, 3, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.002344219123474507*Abs[m["e"]]^2*
      Abs[(1.*m["\[Tau]"] - 2.*mbar["Bs"]*RV["Bs"])^2/
        (mbar["Bs"]^4*RV["Bs"]^4)] + (0.009376876493898028*
       Abs[(1.*m["\[Tau]"] - 2.*mbar["Bs"]*RV["Bs"])^2/RV["Bs"]^2])/
      Abs[mbar["Bs"]]^2
 
\[Beta]2["Bs", 1, 3, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[2, 3]] - (Conjugate[uL[2, 3]]*m["\[Tau]"])/
           (2*mbar["Bs"]*RV["Bs"])]^2*Abs[uL[3, 1]]^2 + 
      Abs[Conjugate[uL[2, 3]] - (Conjugate[uR[2, 3]]*m["\[Tau]"])/
           (2*mbar["Bs"]*RV["Bs"])]^2*Abs[uR[3, 1]]^2)/2
 
\[Beta]2["Bs", 1, 3, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] - Sin[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[-(Cos[Subscript[\[Theta], 13*"L"]]*m["\[Tau]"]*
             Sin[Subscript[\[Theta], 23*"L"]])/
           (2*E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], 
                "L"] + Subscript[\[Chi]2, "L"]))*mbar["Bs"]*RV["Bs"]) + 
          (Cos[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
              Subscript[\[Chi]2, "R"]))]^2 + 
      Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 23*"L"]])/
           E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
              Subscript[\[Chi]2, "L"])) - (Cos[Subscript[\[Theta], 13*"R"]]*
            m["\[Tau]"]*Sin[Subscript[\[Theta], 23*"R"]])/
           (2*E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], 
                "R"] + Subscript[\[Chi]2, "R"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] - Sin[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2)/2
 
\[Beta]2["Bs", 2, 1, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (0.20371310533962786*Abs[((0.7251648098161689*m["e"] - 
            1.*mbar["Bs"]*RV["Bs"])^2*(0.3447492164758735*m["\[Mu]"] - 
            1.*mbar["Bs"]*RV["Bs"])^2)/RV["Bs"]^4] + 
      0.20371310533962791*Abs[((0.3447492164758735*m["e"] - 
            1.*mbar["Bs"]*RV["Bs"])^2*(0.7251648098161688*m["\[Mu]"] - 
            1.*mbar["Bs"]*RV["Bs"])^2)/RV["Bs"]^4])/Abs[mbar["Bs"]]^4
 
\[Beta]2["Bs", 2, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (0.10712538871064461*
       Abs[((1.*E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*m["e"] - 1.3789968659034937*
             E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*mbar["Bs"]*RV["Bs"])^2*
          (0.3447492164758735*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + 
                Subscript[\[CurlyPhi]1, "L"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*mbar["Bs"]*RV["Bs"])^2)/RV["Bs"]^4] + 
      0.024211714530627627*
       Abs[((0.7251648098161688*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + 
                Subscript[\[CurlyPhi]1, "R"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*mbar["Bs"]*RV["Bs"])^2*
          (1.*E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*m["e"] - 2.9006592392646753*
             E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*mbar["Bs"]*RV["Bs"])^2)/RV["Bs"]^4])/Abs[mbar["Bs"]]^4
 
\[Beta]2["Bs", 2, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 2]]^2*Abs[uR[2, 1]]^2 + 
      Abs[uL[2, 1]]^2*Abs[uR[3, 2]]^2)/2
 
\[Beta]2["Bs", 2, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*"L"]] + 
          (Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/E^(I*Subscript[\[Delta], "L"])]^
        2*Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
           Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] + Cos[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2 + 
      Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
           Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] + Cos[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             12*"R"]] + (Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 
              23*"R"]])/E^(I*Subscript[\[Delta], "R"])]^2)/2
 
\[Beta]2["Bs", 2, 2, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    0.2247884495029505*Abs[(0. + 1.*I) - ((0. + 0.3447492164758735*I)*
           m["\[Mu]"])/(mbar["Bs"]*RV["Bs"])]^2 + 0.014049278093934406*
      Abs[m["\[Mu]"]]^2*Abs[(1.*m["\[Mu]"] - 1.378996865903494*mbar["Bs"]*
           RV["Bs"])^2/(mbar["Bs"]^4*RV["Bs"]^4)]
 
\[Beta]2["Bs", 2, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.014049278093934406*Abs[m["\[Mu]"]]^2*
      Abs[(1.*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
               "R"]))*m["\[Mu]"] - 1.378996865903494*
           E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
               "L"]))*mbar["Bs"]*RV["Bs"])^2/(mbar["Bs"]^4*RV["Bs"]^4)] + 
     (0.026716561804279358*
       Abs[(1.*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                "L"]))*m["\[Mu]"] - 2.9006592392646753*
            E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                "R"]))*mbar["Bs"]*RV["Bs"])^2/RV["Bs"]^2])/Abs[mbar["Bs"]]^2
 
\[Beta]2["Bs", 2, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = (Abs[uL[3, 2]]^2*Abs[uR[2, 2]]^2 + 
      Abs[uL[2, 2]]^2*Abs[uR[3, 2]]^2)/2
 
\[Beta]2["Bs", 2, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 23*"L"]] - 
          (Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/E^(I*Subscript[\[Delta], "L"])]^
        2*Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
           Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] + Cos[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2 + 
      Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
           Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] + Cos[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 
             23*"R"]] - (Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 
              23*"R"]])/E^(I*Subscript[\[Delta], "R"])]^2)/2
 
\[Beta]2["Bs", 2, 3, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (0.07149844515742151*Abs[((0.3447492164758735*m["\[Mu]"] - 
            1.*mbar["Bs"]*RV["Bs"])^2*(0.5*m["\[Tau]"] - 1.*mbar["Bs"]*
             RV["Bs"])^2)/RV["Bs"]^4] + 0.0339909391818294*
       Abs[((0.7251648098161688*m["\[Mu]"] - 1.*mbar["Bs"]*RV["Bs"])^2*
          (0.5*m["\[Tau]"] - 1.*mbar["Bs"]*RV["Bs"])^2)/RV["Bs"]^4])/
     Abs[mbar["Bs"]]^4
 
\[Beta]2["Bs", 2, 3, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (0.008497734795457351*Abs[((1.*m["\[Tau]"] - 2.*mbar["Bs"]*RV["Bs"])^2*
          (0.7251648098161688*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + 
                Subscript[\[CurlyPhi]1, "R"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                 "L"]))*mbar["Bs"]*RV["Bs"])^2)/RV["Bs"]^4] + 
      0.01787461128935538*Abs[((1.*m["\[Tau]"] - 2.*mbar["Bs"]*RV["Bs"])^2*
          (0.3447492164758735*E^(I*(2*Subscript[\[CurlyPhi]0, "R"] + 
                Subscript[\[CurlyPhi]1, "L"]))*m["\[Mu]"] - 
            1.*E^(I*(2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                 "R"]))*mbar["Bs"]*RV["Bs"])^2)/RV["Bs"]^4])/Abs[mbar["Bs"]]^4
 
\[Beta]2["Bs", 2, 3, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[2, 3]] - (Conjugate[uL[2, 3]]*m["\[Tau]"])/
           (2*mbar["Bs"]*RV["Bs"])]^2*Abs[uL[3, 2]]^2 + 
      Abs[Conjugate[uL[2, 3]] - (Conjugate[uR[2, 3]]*m["\[Tau]"])/
           (2*mbar["Bs"]*RV["Bs"])]^2*Abs[uR[3, 2]]^2)/2
 
\[Beta]2["Bs", 2, 3, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
           Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 
             13*"L"]] + Cos[Subscript[\[Theta], 12*"L"]]*
           Sin[Subscript[\[Theta], 23*"L"]]]^2*
       Abs[-(Cos[Subscript[\[Theta], 13*"L"]]*m["\[Tau]"]*
             Sin[Subscript[\[Theta], 23*"L"]])/
           (2*E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], 
                "L"] + Subscript[\[Chi]2, "L"]))*mbar["Bs"]*RV["Bs"]) + 
          (Cos[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
              Subscript[\[Chi]2, "R"]))]^2 + 
      Abs[E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
           Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 
             13*"R"]] + Cos[Subscript[\[Theta], 12*"R"]]*
           Sin[Subscript[\[Theta], 23*"R"]]]^2*
       Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 
              23*"L"]])/E^(I*(Subscript[\[Delta], "L"] + Subscript[
               \[CurlyEpsilon], "L"] + Subscript[\[Chi]2, "L"])) - 
          (Cos[Subscript[\[Theta], 13*"R"]]*m["\[Tau]"]*
            Sin[Subscript[\[Theta], 23*"R"]])/
           (2*E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], 
                "R"] + Subscript[\[Chi]2, "R"]))*mbar["Bs"]*RV["Bs"])]^2)/2
 
\[Beta]2["Bs", 3, 1, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 0.
 
\[Beta]2["Bs", 3, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 0.
 
\[Beta]2["Bs", 3, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[uL[2, 1]]^2*Abs[-(m["\[Tau]"]*uL[3, 3])/(2*mbar["Bs"]*RV["Bs"]) + 
          uR[3, 3]]^2 + Abs[uR[2, 1]]^2*
       Abs[uL[3, 3] - (m["\[Tau]"]*uR[3, 3])/(2*mbar["Bs"]*RV["Bs"])]^2)/2
 
\[Beta]2["Bs", 3, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[(Cos[Subscript[\[Theta], 13*"R"]]*Cos[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
               "R"])) - (Cos[Subscript[\[Theta], 13*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
                "L"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
             12*"L"]] + (Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 
              23*"L"]])/E^(I*Subscript[\[Delta], "L"])]^2 + 
      Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 23*"L"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
               "L"])) - (Cos[Subscript[\[Theta], 13*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
                "R"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
             12*"R"]] + (Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 
              23*"R"]])/E^(I*Subscript[\[Delta], "R"])]^2)/2
 
\[Beta]2["Bs", 3, 2, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 0.
 
\[Beta]2["Bs", 3, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 0.
 
\[Beta]2["Bs", 3, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[uL[2, 2]]^2*Abs[-(m["\[Tau]"]*uL[3, 3])/(2*mbar["Bs"]*RV["Bs"]) + 
          uR[3, 3]]^2 + Abs[uR[2, 2]]^2*
       Abs[uL[3, 3] - (m["\[Tau]"]*uR[3, 3])/(2*mbar["Bs"]*RV["Bs"])]^2)/2
 
\[Beta]2["Bs", 3, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[(Cos[Subscript[\[Theta], 13*"R"]]*Cos[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
               "R"])) - (Cos[Subscript[\[Theta], 13*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
                "L"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 
             23*"L"]] - (Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 
              23*"L"]])/E^(I*Subscript[\[Delta], "L"])]^2 + 
      Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 23*"L"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
               "L"])) - (Cos[Subscript[\[Theta], 13*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
                "R"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 
             23*"R"]] - (Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 
              23*"R"]])/E^(I*Subscript[\[Delta], "R"])]^2)/2
 
\[Beta]2["Bs", 3, 3, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 0.
 
\[Beta]2["Bs", 3, 3, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 0.
 
\[Beta]2["Bs", 3, 3, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uL[2, 3]] - (Conjugate[uR[2, 3]]*m["\[Tau]"])/
           (2*mbar["Bs"]*RV["Bs"])]^2*
       Abs[-(m["\[Tau]"]*uL[3, 3])/(2*mbar["Bs"]*RV["Bs"]) + uR[3, 3]]^2 + 
      Abs[Conjugate[uR[2, 3]] - (Conjugate[uL[2, 3]]*m["\[Tau]"])/
           (2*mbar["Bs"]*RV["Bs"])]^2*
       Abs[uL[3, 3] - (m["\[Tau]"]*uR[3, 3])/(2*mbar["Bs"]*RV["Bs"])]^2)/2
 
\[Beta]2["Bs", 3, 3, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Cos[Subscript[\[Theta], 23*"L"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
               "L"])) - (Cos[Subscript[\[Theta], 13*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
                "R"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[-(Cos[Subscript[\[Theta], 13*"L"]]*m["\[Tau]"]*
             Sin[Subscript[\[Theta], 23*"L"]])/
           (2*E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], 
                "L"] + Subscript[\[Chi]2, "L"]))*mbar["Bs"]*RV["Bs"]) + 
          (Cos[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
              Subscript[\[Chi]2, "R"]))]^2 + 
      Abs[(Cos[Subscript[\[Theta], 13*"R"]]*Cos[Subscript[\[Theta], 23*"R"]])/
           E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
               "R"])) - (Cos[Subscript[\[Theta], 13*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]]*m["\[Tau]"])/
           (2*E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
                "L"]))*mbar["Bs"]*RV["Bs"])]^2*
       Abs[(Cos[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 
              23*"L"]])/E^(I*(Subscript[\[Delta], "L"] + Subscript[
               \[CurlyEpsilon], "L"] + Subscript[\[Chi]2, "L"])) - 
          (Cos[Subscript[\[Theta], 13*"R"]]*m["\[Tau]"]*
            Sin[Subscript[\[Theta], 23*"R"]])/
           (2*E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], 
                "R"] + Subscript[\[Chi]2, "R"]))*mbar["Bs"]*RV["Bs"])]^2)/2
 
\[Beta]2["K0L", 1, 1, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (Abs[(0. + 5.551115123125783*^-17*I) + 
         m["e"]*((0. + 0.25824781755683757*I)/(mbar["K0"]*RV["K0"]) + 
           m["e"]*((0. - 0.06034312365542038*I)/(mbar["K0"]^2*RV["K0"]^2) + 
             (0. + 0.06034312365542037*I)/
              (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
               RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)) - 
           (0. + 0.25824781755683757*I)/
            (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]))]^2 + 
      Abs[(0. - 5.551115123125783*^-17*I) + 
         m["e"]*((0. + 0.25824781755683757*I)/(mbar["K0"]*RV["K0"]) + 
           m["e"]*((0. - 0.06034312365542037*I)/(mbar["K0"]^2*RV["K0"]^2) + 
             (0. + 0.06034312365542038*I)/
              (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
               RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)) - 
           (0. + 0.25824781755683757*I)/
            (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]))]^2)/4
 
\[Beta]2["K0L", 1, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (Abs[(0. + 8.326672684688674*^-17*I)/E^(I*(Subscript[\[CurlyPhi]0, "L"] - 
             Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[CurlyPhi]1, "R"])) + 
         (m["e"]*(mbar["K0"]*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["K0"]*RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             ((0. - 0.25824781755683757*I)*mbar["K0"]*RV["K0"] + 
              (0. + 0.25824781755683757*I)*mbar[
                "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*RV[
                "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]) + 
            (0. + 0.06034312365542037*I)*E^(I*(Subscript[\[CurlyPhi]0, "L"] - 
                Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "L"] - 
                Subscript[\[CurlyPhi]1, "R"]))*m["e"]*
             (1.*mbar["K0"]^2*RV["K0"]^2 - 1.0000000000000002*
               mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
               RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)))/
          (mbar["K0"]^2*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
           RV["K0"]^2*RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)]^2 + 
      Abs[(0. - 8.326672684688674*^-17*I)*
          E^(I*(Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "L"] - Subscript[\[CurlyPhi]1, "R"])) + 
         (m["e"]*(mbar["K0"]*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["K0"]*RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             ((0. - 0.25824781755683757*I)*mbar["K0"]*RV["K0"] + 
              (0. + 0.25824781755683757*I)*mbar[
                "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*RV[
                "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]) + 
            ((0. + 0.06034312365542038*I)*m["e"]*(1.*mbar["K0"]^2*
                RV["K0"]^2 - 0.9999999999999998*
                mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
                RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2))/
             E^(I*(Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]0, 
                 "R"] + Subscript[\[CurlyPhi]1, "L"] - Subscript[
                 \[CurlyPhi]1, "R"]))))/(mbar["K0"]^2*
           mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*RV["K0"]^2*
           RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)]^2)/4
 
\[Beta]2["K0L", 1, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[2, 1]]*uL[1, 1] + Conjugate[uR[1, 1]]*uL[2, 1]]^2 + 
      Abs[Conjugate[uL[2, 1]]*uR[1, 1] + Conjugate[uL[1, 1]]*uR[2, 1]]^2)/4
 
\[Beta]2["K0L", 1, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 12*"R"]]*
          Cos[Subscript[\[Theta], 13*"R"]]*(Cos[Subscript[\[Theta], 23*"L"]]*
            Sin[Subscript[\[Theta], 12*"L"]] + E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]) + 
         Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]]*
          (Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
              12*"R"]] + (Cos[Subscript[\[Theta], 12*"R"]]*
             Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 23*
                "R"]])/E^(I*Subscript[\[Delta], "R"]))]^2 + 
      Abs[Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]]*
          (Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
              12*"L"]] + (Cos[Subscript[\[Theta], 12*"L"]]*
             Sin[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 23*
                "L"]])/E^(I*Subscript[\[Delta], "L"])) + 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 12*"L"]]*
          Cos[Subscript[\[Theta], 13*"L"]]*(Cos[Subscript[\[Theta], 23*"R"]]*
            Sin[Subscript[\[Theta], 12*"R"]] + E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])]^2)/4
 
\[Beta]2["K0L", 1, 2, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (Abs[m["e"]]^2*Abs[(0. + 0.08741150105010506*I)/(mbar["K0"]*RV["K0"]) - 
          (0. + 0.08741150105010506*I)/
           (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
            RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2 + 
      Abs[m["\[Mu]"]]^2*Abs[((0. + 0.06338774453474529*I)*m["e"])/
           (mbar["K0"]^2*RV["K0"]^2) - (0. + 0.08741150105010506*I)/
           (mbar["K0"]*RV["K0"]) - ((0. + 0.06338774453474527*I)*m["e"])/
           (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
            RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2) + 
          (0. + 0.08741150105010506*I)/
           (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
            RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2)/4
 
\[Beta]2["K0L", 1, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.25*(Abs[(0. + 5.551115123125783*^-17*I)*
          E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
              "L"])) + ((0. + 0.08741150105010506*I)*
           E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
               "R"]))*m["e"]*(1.*mbar["K0"]*RV["K0"] - 
            1.*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]))/
          (mbar["K0"]*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*RV["K0"]*
           RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2 + 
      0.004018006157202129*Abs[m["\[Mu]"]]^2*
       Abs[(E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
                "R"]))*mbar["K0"]*mbar[
             "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*RV["K0"]*
            RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
            (-1.3789968659034941*mbar["K0"]*RV["K0"] + 1.3789968659034941*
              mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
              RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]) + 
           E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
                "L"]))*m["e"]*(1.*mbar["K0"]^2*RV["K0"]^2 - 
             1.*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
              RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2))/
          (mbar["K0"]^2*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
           RV["K0"]^2*RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)]^2)
 
\[Beta]2["K0L", 1, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[2, 2]]*uL[1, 1] + Conjugate[uR[1, 2]]*uL[2, 1]]^2 + 
      Abs[Conjugate[uL[2, 2]]*uR[1, 1] + Conjugate[uL[1, 2]]*uR[2, 1]]^2)/4
 
\[Beta]2["K0L", 1, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]]*
          (Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 
              23*"L"]] - (Sin[Subscript[\[Theta], 12*"L"]]*
             Sin[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 23*
                "L"]])/E^(I*Subscript[\[Delta], "L"])) - 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 12*"L"]]*(Cos[Subscript[\[Theta], 23*"R"]]*
            Sin[Subscript[\[Theta], 12*"R"]] + E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])]^2 + 
      Abs[E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          (Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
              12*"L"]] + E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])*Sin[Subscript[\[Theta], 
            12*"R"]] - Cos[Subscript[\[Theta], 12*"L"]]*
          Cos[Subscript[\[Theta], 13*"L"]]*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - 
           (Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
             Sin[Subscript[\[Theta], 23*"R"]])/
            E^(I*Subscript[\[Delta], "R"]))]^2)/4
 
\[Beta]2["K0L", 2, 1, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (Abs[m["e"]]^2*Abs[(0. + 0.08741150105010506*I)/(mbar["K0"]*RV["K0"]) - 
          (0. + 0.08741150105010506*I)/
           (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
            RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2 + 
      Abs[m["\[Mu]"]]^2*Abs[((0. + 0.06338774453474527*I)*m["e"])/
           (mbar["K0"]^2*RV["K0"]^2) - (0. + 0.08741150105010506*I)/
           (mbar["K0"]*RV["K0"]) - ((0. + 0.06338774453474529*I)*m["e"])/
           (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
            RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2) + 
          (0. + 0.08741150105010506*I)/
           (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
            RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2)/4
 
\[Beta]2["K0L", 2, 1, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    0.0019101926289581295*Abs[m["e"]]^2*
      Abs[1./(mbar["K0"]*RV["K0"]) - 
         1./(mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
           RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2 + 
     0.0010045015393005327*Abs[m["\[Mu]"]]^2*
      Abs[(E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, 
               "L"]))*mbar["K0"]*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
           RV["K0"]*RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
           (-1.3789968659034937*mbar["K0"]*RV["K0"] + 1.3789968659034937*
             mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]) + 
          E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
           m["e"]*(1.*mbar["K0"]^2*RV["K0"]^2 - 0.9999999999999998*
             mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
             RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2))/
         (mbar["K0"]^2*mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
          RV["K0"]^2*RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)]^2
 
\[Beta]2["K0L", 2, 1, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[2, 1]]*uL[1, 2] + Conjugate[uR[1, 1]]*uL[2, 2]]^2 + 
      Abs[Conjugate[uL[2, 1]]*uR[1, 2] + Conjugate[uL[1, 1]]*uR[2, 2]]^2)/4
 
\[Beta]2["K0L", 2, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 12*"R"]]*
          Cos[Subscript[\[Theta], 13*"R"]]*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]) - 
         Cos[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]]*
          (Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
              12*"R"]] + (Cos[Subscript[\[Theta], 12*"R"]]*
             Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 23*
                "R"]])/E^(I*Subscript[\[Delta], "R"]))]^2 + 
      Abs[Cos[Subscript[\[Theta], 13*"R"]]*(Cos[Subscript[\[Theta], 23*"L"]]*
            Sin[Subscript[\[Theta], 12*"L"]] + 
           (Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
             Sin[Subscript[\[Theta], 23*"L"]])/
            E^(I*Subscript[\[Delta], "L"]))*Sin[Subscript[\[Theta], 
            12*"R"]] - E^(I*(Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[CurlyEpsilon], "R"]))*Cos[Subscript[\[Theta], 
            12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]]*
          (Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 
              23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])]^2)/4
 
\[Beta]2["K0L", 2, 2, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (Abs[m["\[Mu]"]]^2*Abs[(0. + 0.09182186075753421*I)/
          (mbar["K0"]*RV["K0"]) - (0. + 0.09182186075753421*I)/
          (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
           RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2)/2
 
\[Beta]2["K0L", 2, 2, neglectLightLeptMasses -> False, 
     parametrization -> 
      {{{0.3781492963299588*E^(I*(Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"])), 0., 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*(Pi/2 - Subscript[\[Delta], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (0. - 0.9257446244430247*E^(I*Subscript[\[Delta], "L"])), 0., 
         0.3781492963299588*E^(I*(Pi/2 + Subscript[\[Chi]2, "R"]))}, 
        {0., -1./E^(I*(Pi/2 - 2*Subscript[\[CurlyPhi]0, "L"] + 
             Subscript[\[CurlyPhi]1, "L"] + Subscript[\[Chi]2, "R"])), 0.}}, 
       {{0.2607333472413124*E^(I*(Subscript[\[CurlyPhi]0, "R"] + 
             Subscript[\[CurlyPhi]1, "R"])), 0.2738886853288836*
          E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"])), 0.9257446244430247*
          E^(I*Subscript[\[Chi]2, "R"])}, 
        {E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (0. - 0.638299467866969*E^(I*Subscript[\[Delta], "R"])), 
         E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"]))*(0. - 0.6705049582261872*
            E^(I*Subscript[\[Delta], "R"])), 0.3781492963299588*
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] + Subscript[\[Chi]2, "R"]))}, 
        {0.7242871743701426/E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + 
             Subscript[\[Delta], "R"] - Subscript[\[CurlyPhi]0, "R"] - 
             Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), -0.689498432951747/
          E^(I*((Pi - 2*Subscript[\[Delta], "R"])/2 + Subscript[\[Delta], 
              "R"] - 2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, 
              "R"] + Subscript[\[Chi]2, "R"])), 0.}}}] = 
    (Abs[m["\[Mu]"]]^2*Abs[(0. + 0.09182186075753421*I)/
          (mbar["K0"]*RV["K0"]) - (0. + 0.09182186075753421*I)/
          (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
           RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2)/2
 
\[Beta]2["K0L", 2, 2, neglectLightLeptMasses -> True, 
     parametrization -> None] = 
    (Abs[Conjugate[uR[2, 2]]*uL[1, 2] + Conjugate[uR[1, 2]]*uL[2, 2]]^2 + 
      Abs[Conjugate[uL[2, 2]]*uR[1, 2] + Conjugate[uL[1, 2]]*uR[2, 2]]^2)/4
 
\[Beta]2["K0L", 2, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          (Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 
              23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])*Sin[Subscript[\[Theta], 
            12*"R"]] + Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 12*"L"]]*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - 
           (Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
             Sin[Subscript[\[Theta], 23*"R"]])/
            E^(I*Subscript[\[Delta], "R"]))]^2 + 
      Abs[Cos[Subscript[\[Theta], 13*"R"]]*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - 
           (Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
             Sin[Subscript[\[Theta], 23*"L"]])/
            E^(I*Subscript[\[Delta], "L"]))*Sin[Subscript[\[Theta], 
            12*"R"]] + E^(I*(Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[CurlyEpsilon], "R"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]]*
          (Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 
              23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])]^2)/4
 
\[Beta]2["K0S", 1, 1, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (Abs[(0. - 0.482744989243363*I) + m["e"]*((0. + 0.25824781755683757*I)/
            (mbar["K0"]*RV["K0"]) + m["e"]*((0. - 0.06034312365542038*I)/
              (mbar["K0"]^2*RV["K0"]^2) - (0. + 0.06034312365542037*I)/
              (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
               RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)) + 
           (0. + 0.25824781755683757*I)/
            (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]))]^2 + 
      Abs[(0. - 0.482744989243363*I) + m["e"]*((0. + 0.25824781755683757*I)/
            (mbar["K0"]*RV["K0"]) + m["e"]*((0. - 0.06034312365542037*I)/
              (mbar["K0"]^2*RV["K0"]^2) - (0. + 0.06034312365542038*I)/
              (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
               RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2)) + 
           (0. + 0.25824781755683757*I)/
            (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]))]^2)/4
 
\[Beta]2["K0S", 1, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 12*"R"]]*
          Cos[Subscript[\[Theta], 13*"R"]]*(Cos[Subscript[\[Theta], 23*"L"]]*
            Sin[Subscript[\[Theta], 12*"L"]] + E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]) - 
         Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]]*
          (Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
              12*"R"]] + (Cos[Subscript[\[Theta], 12*"R"]]*
             Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 23*
                "R"]])/E^(I*Subscript[\[Delta], "R"]))]^2 + 
      Abs[Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]]*
          (Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
              12*"L"]] + (Cos[Subscript[\[Theta], 12*"L"]]*
             Sin[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 23*
                "L"]])/E^(I*Subscript[\[Delta], "L"])) - 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 12*"L"]]*
          Cos[Subscript[\[Theta], 13*"L"]]*(Cos[Subscript[\[Theta], 23*"R"]]*
            Sin[Subscript[\[Theta], 12*"R"]] + E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])]^2)/4
 
\[Beta]2["K0S", 1, 2, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (Abs[(0. + 0.5071019562779622*I) + m["e"]*((0. - 0.08741150105010506*I)/
            (mbar["K0"]*RV["K0"]) - (0. + 0.08741150105010506*I)/
            (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]))]^2 + 
      Abs[m["\[Mu]"]]^2*Abs[((0. + 0.06338774453474529*I)*m["e"])/
           (mbar["K0"]^2*RV["K0"]^2) - (0. + 0.08741150105010506*I)/
           (mbar["K0"]*RV["K0"]) + ((0. + 0.06338774453474527*I)*m["e"])/
           (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
            RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2) - 
          (0. + 0.08741150105010506*I)/
           (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
            RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2)/4
 
\[Beta]2["K0S", 1, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]]*
          (Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 
              23*"L"]] - (Sin[Subscript[\[Theta], 12*"L"]]*
             Sin[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 23*
                "L"]])/E^(I*Subscript[\[Delta], "L"])) + 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 12*"L"]]*(Cos[Subscript[\[Theta], 23*"R"]]*
            Sin[Subscript[\[Theta], 12*"R"]] + E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])]^2 + 
      Abs[E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          (Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 
              12*"L"]] + E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])*Sin[Subscript[\[Theta], 
            12*"R"]] + Cos[Subscript[\[Theta], 12*"L"]]*
          Cos[Subscript[\[Theta], 13*"L"]]*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - 
           (Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
             Sin[Subscript[\[Theta], 23*"R"]])/
            E^(I*Subscript[\[Delta], "R"]))]^2)/4
 
\[Beta]2["K0S", 2, 1, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (Abs[(0. + 0.5071019562779622*I) + m["e"]*((0. - 0.08741150105010506*I)/
            (mbar["K0"]*RV["K0"]) - (0. + 0.08741150105010506*I)/
            (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
             RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]))]^2 + 
      Abs[m["\[Mu]"]]^2*Abs[((0. + 0.06338774453474527*I)*m["e"])/
           (mbar["K0"]^2*RV["K0"]^2) - (0. + 0.08741150105010506*I)/
           (mbar["K0"]*RV["K0"]) + ((0. + 0.06338774453474529*I)*m["e"])/
           (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2*
            RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]^2) - 
          (0. + 0.08741150105010506*I)/
           (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
            RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2)/4
 
\[Beta]2["K0S", 2, 1, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 12*"R"]]*
          Cos[Subscript[\[Theta], 13*"R"]]*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]) + 
         Cos[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]]*
          (Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 
              12*"R"]] + (Cos[Subscript[\[Theta], 12*"R"]]*
             Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 23*
                "R"]])/E^(I*Subscript[\[Delta], "R"]))]^2 + 
      Abs[Cos[Subscript[\[Theta], 13*"R"]]*(Cos[Subscript[\[Theta], 23*"L"]]*
            Sin[Subscript[\[Theta], 12*"L"]] + 
           (Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
             Sin[Subscript[\[Theta], 23*"L"]])/
            E^(I*Subscript[\[Delta], "L"]))*Sin[Subscript[\[Theta], 
            12*"R"]] + E^(I*(Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[CurlyEpsilon], "R"]))*Cos[Subscript[\[Theta], 
            12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]]*
          (Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 
              23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])]^2)/4
 
\[Beta]2["K0S", 2, 2, neglectLightLeptMasses -> False, 
     parametrization -> {{{0.3781492963299588, 0., 0.9257446244430247}, 
        {0. - 0.9257446244430247*I, 0. + 0.*I, 0. + 0.3781492963299588*I}, 
        {0. + 0.*I, 0. + 1.*I, 0. + 0.*I}}, 
       {{0.2607333472413124, 0.2738886853288836, 0.9257446244430247}, 
        {0. - 0.638299467866969*I, 0. - 0.6705049582261872*I, 
         0. + 0.3781492963299588*I}, {0. - 0.7242871743701426*I, 
         0. + 0.689498432951747*I, 0. + 0.*I}}}] = 
    (Abs[m["\[Mu]"]]^2*Abs[(0. + 0.09182186075753421*I)/
          (mbar["K0"]*RV["K0"]) + (0. + 0.09182186075753421*I)/
          (mbar["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"]*
           RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"])]^2)/2
 
\[Beta]2["K0S", 2, 2, neglectLightLeptMasses -> True, 
     parametrization -> 
      {{{E^(I*(Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 13*"L"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, 
              "L"] - Subscript[\[Chi]1, "L"]))*Cos[Subscript[\[Theta], 
            13*"L"]]*Sin[Subscript[\[Theta], 12*"L"]], 
         E^(I*Subscript[\[Chi]2, "L"])*Sin[Subscript[\[Theta], 13*"L"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyPhi]0, 
              "L"] + Subscript[\[CurlyPhi]1, "L"]))*
          (-(Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 12*
                "L"]]) - E^(I*Subscript[\[Delta], "L"])*
            Cos[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + 2*Subscript[\[CurlyPhi]0, 
               "L"] - Subscript[\[CurlyPhi]1, "L"] - Subscript[\[Chi]1, 
              "L"]))*(Cos[Subscript[\[Theta], 12*"L"]]*
            Cos[Subscript[\[Theta], 23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]]), 
         E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] + 
             Subscript[\[Chi]2, "L"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 23*"L"]]}, 
        {(-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 12*"L"]]*
             Cos[Subscript[\[Theta], 23*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) + Sin[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             Subscript[\[CurlyPhi]0, "L"] - Subscript[\[CurlyPhi]1, "L"] - 
             Subscript[\[Chi]1, "L"] + Subscript[\[Chi]2, "L"])), 
         (-(E^(I*Subscript[\[Delta], "L"])*Cos[Subscript[\[Theta], 23*"L"]]*
             Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*
                "L"]]) - Cos[Subscript[\[Theta], 12*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[Delta], "L"] + Subscript[\[CurlyEpsilon], "L"] - 
             2*Subscript[\[CurlyPhi]0, "L"] + Subscript[\[CurlyPhi]1, "L"] + 
             Subscript[\[Chi]2, "L"])), (Cos[Subscript[\[Theta], 13*"L"]]*
           Cos[Subscript[\[Theta], 23*"L"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "L"] - Subscript[\[Chi]1, 
              "L"]))}}, 
       {{E^(I*(Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 13*"R"]], 
         E^(I*(2*Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, 
              "R"] - Subscript[\[Chi]1, "R"]))*Cos[Subscript[\[Theta], 
            13*"R"]]*Sin[Subscript[\[Theta], 12*"R"]], 
         E^(I*Subscript[\[Chi]2, "R"])*Sin[Subscript[\[Theta], 13*"R"]]}, 
        {E^(I*(Subscript[\[CurlyEpsilon], "R"] + Subscript[\[CurlyPhi]0, 
              "R"] + Subscript[\[CurlyPhi]1, "R"]))*
          (-(Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 12*
                "R"]]) - E^(I*Subscript[\[Delta], "R"])*
            Cos[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[CurlyEpsilon], "R"] + 2*Subscript[\[CurlyPhi]0, 
               "R"] - Subscript[\[CurlyPhi]1, "R"] - Subscript[\[Chi]1, 
              "R"]))*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]]), 
         E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] + 
             Subscript[\[Chi]2, "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          Sin[Subscript[\[Theta], 23*"R"]]}, 
        {(-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 12*"R"]]*
             Cos[Subscript[\[Theta], 23*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) + Sin[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             Subscript[\[CurlyPhi]0, "R"] - Subscript[\[CurlyPhi]1, "R"] - 
             Subscript[\[Chi]1, "R"] + Subscript[\[Chi]2, "R"])), 
         (-(E^(I*Subscript[\[Delta], "R"])*Cos[Subscript[\[Theta], 23*"R"]]*
             Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*
                "R"]]) - Cos[Subscript[\[Theta], 12*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[Delta], "R"] + Subscript[\[CurlyEpsilon], "R"] - 
             2*Subscript[\[CurlyPhi]0, "R"] + Subscript[\[CurlyPhi]1, "R"] + 
             Subscript[\[Chi]2, "R"])), (Cos[Subscript[\[Theta], 13*"R"]]*
           Cos[Subscript[\[Theta], 23*"R"]])/
          E^(I*(Subscript[\[CurlyEpsilon], "R"] - Subscript[\[Chi]1, 
              "R"]))}}}] = 
    (Abs[E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 13*"R"]]*
          (Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 
              23*"L"]] - E^(I*Subscript[\[Delta], "L"])*
            Sin[Subscript[\[Theta], 12*"L"]]*Sin[Subscript[\[Theta], 13*"L"]]*
            Sin[Subscript[\[Theta], 23*"L"]])*Sin[Subscript[\[Theta], 
            12*"R"]] + Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 12*"L"]]*
          (-(Cos[Subscript[\[Theta], 12*"R"]]*Cos[Subscript[\[Theta], 23*
                "R"]]) + (Sin[Subscript[\[Theta], 12*"R"]]*
             Sin[Subscript[\[Theta], 13*"R"]]*Sin[Subscript[\[Theta], 23*
                "R"]])/E^(I*Subscript[\[Delta], "R"]))]^2 + 
      Abs[Cos[Subscript[\[Theta], 13*"R"]]*
          (-(Cos[Subscript[\[Theta], 12*"L"]]*Cos[Subscript[\[Theta], 23*
                "L"]]) + (Sin[Subscript[\[Theta], 12*"L"]]*
             Sin[Subscript[\[Theta], 13*"L"]]*Sin[Subscript[\[Theta], 23*
                "L"]])/E^(I*Subscript[\[Delta], "L"]))*
          Sin[Subscript[\[Theta], 12*"R"]] + 
         E^(I*(Subscript[\[CurlyEpsilon], "L"] + Subscript[\[CurlyEpsilon], 
              "R"]))*Cos[Subscript[\[Theta], 13*"L"]]*
          Sin[Subscript[\[Theta], 12*"L"]]*(Cos[Subscript[\[Theta], 12*"R"]]*
            Cos[Subscript[\[Theta], 23*"R"]] - E^(I*Subscript[\[Delta], "R"])*
            Sin[Subscript[\[Theta], 12*"R"]]*Sin[Subscript[\[Theta], 13*"R"]]*
            Sin[Subscript[\[Theta], 23*"R"]])]^2)/4
 
\[Beta]2[P_String, "ee", opts:OptionsPattern[]] := \[Beta]2[P, 1, 1, opts]
 
\[Beta]2[P_String, "\[Mu]\[Mu]", opts:OptionsPattern[]] := 
    \[Beta]2[P, 2, 2, opts]
 
\[Beta]2[P_String, "\[Tau]\[Tau]", opts:OptionsPattern[]] := 
    \[Beta]2[P, 3, 3, opts]
 
\[Beta]2[P_String, "e\[Mu]", opts:OptionsPattern[]] := 
    \[Beta]2[P, 1, 2, opts] + \[Beta]2[P, 2, 1, opts]
 
\[Beta]2[P_String, "e\[Tau]", opts:OptionsPattern[]] := 
    \[Beta]2[P, 1, 3, opts] + \[Beta]2[P, 3, 1, opts]
 
\[Beta]2[P_String, "\[Mu]\[Tau]", opts:OptionsPattern[]] := 
    \[Beta]2[P, 2, 3, opts] + \[Beta]2[P, 3, 2, opts]
 
\[Beta]2[{P_String, ll_String}, opts:OptionsPattern[]] := 
    \[Beta]2[P, ll, opts]
 
\[Beta]2[P_String, i_Integer, j_Integer, opts:OptionsPattern[]] := 
    (If[ !Keys[Options[\[Beta]2]] === Keys[{opts}], 
      With[{newOpts = Replace[Sort[Join[{opts}, FilterRules[
             Options[\[Beta]2], Except[{opts}]]]], {o___} -> Sequence[o]]}, 
       Return[\[Beta]2[P, i, j, newOpts]]]]; 
     Module[{parametrizace, ampLR2, ampRL2, result, neglRule, Ul, Ur, 
       optsToAmplitudes}, Print["Calculating \[Beta]2[", P, ",", i, ",", j, 
        "] for given Options..."]; Switch[OptionValue[parametrization], None, 
        parametrizace = {}, {(Ul_)?MatrixQ, (Ur_)?MatrixQ}, 
        {Ul, Ur} = OptionValue[parametrization]; parametrizace = 
          {uL[a_, b_] :> Ul[[a,b]], uR[a_, b_] :> Ur[[a,b]]}, _, 
        Print["Invalid Parametrization!!!"]; Return[False]]; 
       optsToAmplitudes = FilterRules[{opts}, Options[ampLR]]; 
       ampLR2 = FullSimplify[Abs[(ampLR[P, i, j, optsToAmplitudes] /. 
             parametrizace)^2], TimeConstraint -> {10, 120}] /. 
         Abs[(X_)*(Y_)] -> Abs[X]*Abs[Y]; If[Ul === (Ur /. swapLR) || 
         OptionValue[parametrization] === None, ampRL2 = ampLR2 /. swapLR, 
        ampRL2 = FullSimplify[Abs[(ampLR[P, i, j, optsToAmplitudes] /. 
               swapLR /. parametrizace)^2], TimeConstraint -> {10, 120}] /. 
          Abs[(X_)*(Y_)] -> Abs[X]*Abs[Y]]; \[Beta]2[P, i, j, opts] = 
        Simplify[(ampLR2 + ampRL2)/2]])
 
\[Beta]2[{P_String, i_Integer, j_Integer}, opts:OptionsPattern[]] := 
    \[Beta]2[P, i, j, opts]
 
Options[\[Beta]2] = {neglectLightLeptMasses -> True, parametrization -> None}
 
Attributes[Subscript] = {NHoldRest}
 
Subscript[q, "B0"] = 1
 
Subscript[q, "Bs"] = 2
 
Subscript[q, "K0"] = 1
 
Subscript[q, "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"] = 2
 
Subscript[OverBar[q], "B0"] = 3
 
Subscript[OverBar[q], "Bs"] = 3
 
Subscript[OverBar[q], "K0"] = 2
 
Subscript[OverBar[q], "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"] = 1
 
OverBar[m] := mbar
 
m /: N[m["B0"], {MachinePrecision, MachinePrecision}] = 
     Quantity[5279.4`5., "Megaelectronvolts"]
 
m /: N[m["Bs"], {MachinePrecision, MachinePrecision}] = 
     Quantity[5367.5`5., "Megaelectronvolts"]
 
m /: N[m["e"], {MachinePrecision, MachinePrecision}] = 
     Quantity[0.51099892`8., "Megaelectronvolts"]
 
m /: N[m["K0L"], {MachinePrecision, MachinePrecision}] = 
     Quantity[497.648`6., "Megaelectronvolts"]
 
m /: N[m["K0S"], {MachinePrecision, MachinePrecision}] = 
     Quantity[497.648`6., "Megaelectronvolts"]
 
m /: N[m["\[Mu]"], {MachinePrecision, MachinePrecision}] = 
     Quantity[105.658369`9., "Megaelectronvolts"]
 
m /: N[m["\[Tau]"], {MachinePrecision, MachinePrecision}] = 
     Quantity[1776.99`6., "Megaelectronvolts"]
 
m /: N[m["b", Quantity[5, "Gigaelectronvolts"]], {MachinePrecision, 
       MachinePrecision}] := 4.18*GeV
 
m /: N[m["d", Quantity[1, "Gigaelectronvolts"]], {MachinePrecision, 
       MachinePrecision}] := 6.3*MeV
 
m /: N[m["d", Quantity[5, "Gigaelectronvolts"]], {MachinePrecision, 
       MachinePrecision}] := 3.8*MeV
 
m /: N[m["s", Quantity[1, "Gigaelectronvolts"]], {MachinePrecision, 
       MachinePrecision}] := 129.5*MeV
 
m /: N[m["s", Quantity[5, "Gigaelectronvolts"]], {MachinePrecision, 
       MachinePrecision}] := 78.3*MeV
 
m /: N[m[P_ /; MemberQ[Join[leptons, mesons], P]], {MachinePrecision, 
       MachinePrecision}] := N[m[P]] = ParticleData[wolframParticleNames[P], 
        "Mass"]*Quantity["SpeedOfLight"]^2
 
MeV = Quantity[1, "Megaelectronvolts"]
 
GeV = Quantity[1, "Gigaelectronvolts"]
 
leptons := {"e", "\[Mu]", "\[Tau]"}
 
mesons := Join[kaonMassEigStates, bottomMesons]
 
kaonMassEigStates := {"K0L", "K0S"}
 
bottomMesons := {"B0", "Bs"}
 
wolframParticleNames = <|"K0L" -> "KLong", "B0" -> "BZero", 
     "Bs" -> {"BSMeson", 0}, "K0S" -> "KShort", "e" -> "Electron", 
     "\[Mu]" -> "Muon", "\[Tau]" -> "TauLepton"|>
 
TraditionalForm[mbar[P_]] ^:= TraditionalForm[
     "\!\(\*OverscriptBox[\(m\), \(_\)]\)"[P]]
 
mbar /: N[mbar["B0"], {MachinePrecision, MachinePrecision}] := 
     m["B0"]^2/(m["b", 5*GeV] + m["d", 5*GeV])
 
mbar /: N[mbar["Bs"], {MachinePrecision, MachinePrecision}] := 
     m["Bs"]^2/(m["b", 5*GeV] + m["s", 5*GeV])
 
mbar /: N[mbar[k_ /; MemberQ[{"K0L", "K0S", "K0", 
          "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"}, k]], 
      {MachinePrecision, MachinePrecision}] := 
     m["K0L"]^2/(m["s", 1*GeV] + m["d", 1*GeV])
 
RV /: N[RV["B0"], {MachinePrecision, MachinePrecision}] := R4V["B"]
 
RV /: N[RV["Bs"], {MachinePrecision, MachinePrecision}] := R4V["B"]
 
RV /: N[RV["K0"], {MachinePrecision, MachinePrecision}] := R4V["K"]
 
RV /: N[RV["K0L"], {MachinePrecision, MachinePrecision}] := R4V["K"]
 
RV /: N[RV["K0S"], {MachinePrecision, MachinePrecision}] := R4V["K"]
 
RV /: N[RV["\!\(\*OverscriptBox[\(K0\), \(_\)]\)"], {MachinePrecision, 
       MachinePrecision}] := R4V["K"]
 
R4V["B"] = 2.1
 
R4V["K"] = 3.47
 
ampLR[P_ /; MemberQ[mesonWeakEigStates, P], i_Integer, j_Integer, 
     OptionsPattern[]] := (uL[Subscript[OverBar[q], P], i] - 
       uR[Subscript[OverBar[q], P], i]*(m[leptons[[i]]]/
         (2*OverBar[m][P]*RV[P])))*(Conjugate[uR[Subscript[q, P], j]] - 
       Conjugate[uL[Subscript[q, P], j]]*(m[leptons[[j]]]/
         (2*OverBar[m][P]*RV[P]))) /. 
     If[OptionValue[neglectLightLeptMasses] == True, 
      {m["e"] -> 0, m["\[Mu]"] -> 0}, {}]
 
ampLR["K0L", i_Integer, j_Integer, opts:OptionsPattern[]] := 
    (ampLR["K0", i, j, opts] + ampLR["\!\(\*OverscriptBox[\(K0\), \(_\)]\)", 
       i, j, opts])/Sqrt[2]
 
ampLR["K0S", i_Integer, j_Integer, opts:OptionsPattern[]] := 
    (ampLR["K0", i, j, opts] - ampLR["\!\(\*OverscriptBox[\(K0\), \(_\)]\)", 
       i, j, opts])/Sqrt[2]
 
Options[ampLR] = {neglectLightLeptMasses -> True}
 
mesonWeakEigStates := Join[kaonWeakEigStates, bottomMesons]
 
kaonWeakEigStates := {"K0", "\!\(\*OverscriptBox[\(K0\), \(_\)]\)"}
 
swapLR = {"L" :> "R", "R" :> "L", ampLR :> ampRL, ampRL :> ampLR, uL :> uR, 
     uR :> uL}
