function [g,grad_g] = compute_max_stress_violation()
%
% This function computes the mean compliance and its sensitivities
% based on the last finite element analysis
%
global FE OPT

% compute the element (centroidal) von Mises stress from the displacement.
% Note that this is not the FEA stress because Me is
% computed with the fully-solid modulus, not with the ersatz modulus.

se = zeros(FE.n_elem, FE.nloads);

for iload=1:FE.nloads
    Ul = FE.U(:,iload);
    for j=1:FE.n_elem
        Ue = Ul(FE.edofMat(j,:));
        % While element matrices Me should be positive semidefinite,
        % numerically they can have very small but negative eigenvalues, hence the
        % von Mises stress can end up being a complex number (with a near-zero
        % complex part). Therefore, we no longer compute se as
        % sqrt(Ue'*Me*Ue), but we compute first the FE stress sigma, and
        % then the von Mises stress as se = sqrt(sigma'*V*sigma).
        B0e = FE.B0e(:,:,j);
        sigma = (FE.material.C*B0e*Ue);
        se(j,iload) = sqrt(sigma'*FE.V*sigma);
    end
end

% Relaxed stress
p = OPT.parameters.penalization_param;
q = OPT.parameters.relaxation_param;
rhomin = FE.material.rho_min;
[re,dredrhof] = relaxdens(OPT.filt_rho_e,p,q,rhomin, 'stdpq');
FE.svm = re.*se;
slim = 1; % Recall for multiple load cases this is a vector

% Compute aggregate stress function

h = FE.svm./slim;
dhds = ones(size(h))./slim;
phi = h;
dphidh = ones(size(h));
P = OPT.parameters.aggregation_parameter;
[g, dgdphi] = smooth_max(reshape(phi,[],1), P, 'p-norm');
dgdphi = reshape(dgdphi,size(h));
g = g - 1;

% Compute pseudo-load
FE.dJdu = zeros(FE.n_global_dof,FE.nloads);
C = FE.material.C;
V = FE.V;
for iload=1:FE.nloads
    Ul = FE.U(:,iload);
    for j=1:FE.n_elem
        Ue = Ul(FE.edofMat(j,:));
        B0e = FE.B0e(:,:,j);
        sigma = (C*B0e*Ue);
        MeUe = B0e'*C*V*sigma;
        FE.dJdu(FE.edofMat(j,:),iload) = FE.dJdu(FE.edofMat(j,:),iload) - ...
            dgdphi(j,iload)*dphidh(j,iload)*dhds(j,iload)* ...
            (FE.svm(j,iload)/se(j,iload)^2)*MeUe;
    end
end


% Solve pseudo analysis to compute adjoint solution
FE_solve('adjoint');

% Compute sensitivities
lTdku = zeros(FE.n_elem,FE.nloads);
for iload=1:FE.nloads
    Ul = FE.U(:,iload);
    Ll = FE.lambda(:,iload);
    for j=1:FE.n_elem
        Ue = Ul(FE.edofMat(j,:));
        Le = Ll(FE.edofMat(j,:));
        dKe = OPT.dpen_rho_e(j)*FE.Ke(:,:,j);
        lTdku(j,iload) = Le'*dKe*Ue;
    end
end
grad_g = zeros(FE.n_elem,1);
for iload=1:FE.nloads
    grad_g = grad_g + dgdphi(:,iload).*dphidh(:,iload).*dhds(:,iload).*dredrhof.* ...
        se(:,iload) + lTdku(:,iload);
end


% Account for filtering in sensitivities
grad_g = OPT.H' * grad_g;
OPT.grad_stress = grad_g;
end