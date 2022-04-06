function image=vec2im(vec,mask)
image=zeros(size(mask));
image(mask)=vec;
end