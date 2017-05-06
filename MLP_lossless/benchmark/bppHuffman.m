function [bpp, entropy]=bppHuffman(array,max_value)
%Calculates the avarage bits per pixel in the array using Huffman Coding
%bpp gives bits per pixel, entropy gives Shannon Entropy (not so important
%I guess)

hstcounts = histcounts(array,0:max_value+1);
prob = hstcounts/sum(hstcounts);
dict = huffmandict(0:max_value,prob);
keys = zeros(1,max_value+1);
for k=1:max_value+1
    key = dict{k,2};
    keys(k)=length(key);
end
bpp=sum(keys.*prob);
entropy=-1*sum(prob(prob>=1e-5).*log(prob(prob>=1e-5)));