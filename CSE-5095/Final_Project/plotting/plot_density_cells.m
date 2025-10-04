function plot_BCs_cells(fig)
%
% Plot the density field into the specified figure
%
global FE OPT

%% Change here whether you want to plot the penalized (i.e., effective) or
%% the unpenalized (i.e., projected) densities.  By default, we plot the
%% effective densities.
%
% For penalized, use OPT.pen_rho_e;
% For unpenalized, use OPT.dv;
%
plot_dens = OPT.pen_rho_e;
%% 2D

F = FE.elem_node.'; % matrix of faces to be sent to patch function
V = FE.coords'; % vertex list to be sent to patch function


figure(fig); cla; hold on

% Plot mesh if requested
p = patch('Faces',F,'Vertices',V);
p.FaceColor = 'none';
p.EdgeColor = 'none';
axis equal;
% Plot stresses
newmap = jet;
% newmap(256,:) = [0.5 0.5 0.5]; % Gray for overflow
colormap(newmap);
s = OPT.pen_rho_e;
p.CData = s;
p.FaceColor = 'flat';
axis off;

end
