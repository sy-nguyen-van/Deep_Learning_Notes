
function fig_out = myfig(fig,F,V,mydata)
fig_out = figure(fig); cla; hold on;
% Plot mesh if requested
p = patch('Faces',F,'Vertices',V);
p.FaceColor = 'none';
p.EdgeColor = 'none';
axis equal;
% Plot stresses
newmap = jet;
% newmap(256,:) = [0.5 0.5 0.5]; % Gray for overflow
colormap(newmap);
s = mydata;
p.CData = s;
p.FaceColor = 'flat';
axis off;
end