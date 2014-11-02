function [coeff,soglie] = genera_rete(K,n)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    coeff = -1+2*rand(K,n);
    soglie = -1+2*rand(K,1);
end

