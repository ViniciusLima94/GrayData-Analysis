function [X,Y] = get_flatmap_coordinates(area)

%gets the coordinates to fill in an area on the flatmap.

%will load the "all_flatmpap_areas.mat" and find the area that is input that
%matches the all_flatmap_areas.mat area.

%Use for plotting on the "Flatmap_outlines.jpg" in the monkey folder

%Usage:
% [X,Y] = get_flatmap_coordinates(area) ;
% note "area" needs to be a string

%Example Usage if want to plot on the flatmap
%open('Flatmap_no_areas.jpg')
%figure
%imshow(Flatmap_no_areas)
%hold on
%area = 'V4'
%[X,Y] = get_flatmap_coordinates(area);
%fill(X(:,1),Y(:,1),'r')
%shg
% %

% 6/2/2016 sh
%First check that all_flatmap_areas is in current directory
if ~exist('all_flatmap_areas.mat','file')
    error('Error, need to have the file "all_flatmap_areas.mat" in current directory')
    
end

load('all_flatmap_areas.mat') ;

switch area
    
    case 'Pi'
        X = Pi(:,1) ; Y = Pi(:,2) ;
    case 'V1'
        X = V1(:,1) ; Y = V1(:,2) ;
        
    case 'V2'
        X = V2(:,1) ; Y = V2(:,2) ;
    case 'V4'
        X = V4(:,1) ; Y = V4(:,2) ;
        
    case 'V4t'
        X = V4t(:,1) ; Y = V4t(:,2) ;
    case 'DP'
        X = DP(:,1) ; Y = DP(:,2) ;
    case 'V3'
        X = V3(:,1) ; Y = V3(:,2) ;
    case 'V3A'
        X = V3A(:,1) ; Y = V3A(:,2) ;
    case 'a5'
        X = a5(:,1) ; Y = a5(:,2) ;
    case 'a7M'
        X = a7m(:,1) ; Y = a7m(:,2) ;
    case 'AIP'
        X = AIP(:,1) ; Y = AIP(:,2) ;
    case 'VIP'
        X = VIP(:,1) ; Y = VIP(:,2) ;
    case 'V6A'
        X = V6A(:,1) ; Y = V6A(:,2) ;
    case 'V6'
        X = V6(:,1) ; Y = V6(:,2) ;
    case 'PPT'   %DONT HAVE PPT WORKED OUT YET  ?maybe eh?
        X = DP(:,1) ; Y = DP(:,2) ;
    case 'MT'
        X = MT(:,1) ; Y = MT(:,2) ;
    case 'TEO'
        X = TEO(:,1) ; Y = TEO(:,2) ;
    case 'TEOm'
        X = TEOm(:,1) ; Y = TEOm(:,2) ;
    case 'PG'
        X = PG(:,1) ; Y = PG(:,2) ;
    case 'TPOC'
        X = STP(:,1) ; Y = STP(:,2) ;
    case 'TPt'
        X = TPt(:,1) ; Y = TPt(:,2) ;
    case 'TPO'
        X = STP(:,1) ; Y = STP(:,2) ;
    case 'TEpv'
        X = TEpv(:,1) ; Y = TEpv(:,2) ;
    case 'TEpd'
        X = TEpd(:,1) ; Y = TEpd(:,2) ;
    case 'FST'
        X = FST(:,1) ; Y = FST(:,2) ;
    case 'MST'
        X= MST(:,1) ; Y = MST(:,2) ;
    case 'a2'
        X = a2(:,1) ; Y = a2(:,2) ;
    case 'a7A'
        X = a7A(:,1) ; Y = a7A(:,2) ;
    case 'a1'
        X = a1(:,1) ; Y = a1(:,2) ;
    case 'PBc'
        X = PBc(:,1) ; Y = PBc(:,2) ;
    case 'a7B'
        X = a7B(:,1) ; Y = a7B(:,2) ;
    case 'LIP'
        X = LIP(:,1) ; Y = LIP(:,2) ;
    case 'MIP'
        X = MIP(:,1) ; Y = MIP(:,2) ;
    case 'PIP'
        X = PIP(:,1) ; Y = PIP(:,2) ;
    case 'a3'
        X = a3(:,1) ; Y = a3(:,2) ;
    case 'a7op'    % NEED TO FIND IN KENNEDY
        X = a7op(:,1) ; Y = a7op(:,2) ;
    case 'F1'
        X = F1(:,1) ; Y = F1(:,2) ;
    case 'SII'
        X = SII(:,1) ; Y = SII(:,2) ;
    case 'F3'
        X = F3(:,1) ; Y = F3(:,2) ;
    case 'a24D'
        X = a24d(:,1) ; Y = a24d(:,2) ;
    case 'F5'
        X = F5(:,1) ; Y = F5(:,2) ;
    case 'F4'
        X = F4(:,1) ; Y = F4(:,2) ;
    case 'F2'
        X = F2(:,1) ; Y = F2(:,2) ;
    case 'a44'
        X = a44(:,1) ; Y = a44(:,2) ;
    case 'OPRO'
        X = OPRO(:,1) ; Y = OPRO(:,2) ;
    case 'ProM'
        X = ProM(:,1) ; Y = ProM(:,2) ;
    case 'a23'
        X = a23(:,1) ; Y = a23(:,2) ;
    case 'a8M'
        X = a8m(:,1) ; Y = a8m(:,2) ;
    case 'a24c'
        X = a24c(:,1) ; Y = a24c(:,2) ;
    case 'a8L'
        X = a8l(:,1) ; Y = a8l(:,2) ;
    case 'F7'
        X = F7(:,1) ; Y = F7(:,2) ;
    case 'F6'
        X = F6(:,1) ; Y = F6(:,2) ;
    case 'a45B'
        X = a45B(:,1) ; Y = a45B(:,2) ;
    case 'a9/46V'
        X = a9_46V(:,1) ; Y = a9_46V(:,2) ;
    case 'a46D'
        X = a46d(:,1) ; Y = a46d(:,2) ;
    case 'a46V'
        X = a46V(:,1) ; Y = a46V(:,2) ;
    case 'a8B'
        X = a8B(:,1) ; Y = a8B(:,2) ;
    case 'a8r'
        X = a8r(:,1) ; Y = a8r(:,2) ;
    case 'a45A'
        X = a45A(:,1) ; Y = a45A(:,2) ;
    case 'a8/32'
        X = a24c(:,1) ; Y = a24c(:,2) ;
    case 'a9/46D'
        X = a9_46D(:,1) ; Y = a9_46D(:,2) ;
    case 'a9'
        X = a9(:,1) ; Y = a9(:,2) ;
    case 'a11'
        X = a11(:,1) ; Y = a11(:,2) ;
    case 'a12'
        X = a12(:,1) ; Y = a12(:,2) ;
    case 'a13'
        X = a13(:,1) ; Y = a13(:,2) ;
    case 'a14'
        X = a14(:,1) ; Y = a14(:,2) ;
    case 'a32'
        X = a32(:,1) ; Y = a32(:,2) ;
    case 'Ins'
        X = Ins(:,1) ; Y = Ins(:,2) ;
    case 'PGa'
        X = PGa(:,1) ; Y= PGa(:,2) ;
    case 'STPc'
        X = STPc(:,1) ; Y = STPc(:,2) ;
    case 'PBr'
        X = PBr(:,1) ; Y = PBr(:,2) ;
    case 'LB'
        X = LB(:,1) ; Y = LB(:,2) ;
    case 'Core'
        X = Core(:,1) ; Y = Core(:,2) ;
    case 'MB'
        X = MB(:,1) ; Y = MB(:,2) ;
        
    otherwise
        display('*** could not find flatmap coordinates for this area!!! ***') ;
end


end