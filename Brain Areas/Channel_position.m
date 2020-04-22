% load('lucy_brainsketch_xy.mat')  %for lucy. For Ethyl it would be load('ethyl_bmf_grid.mat')
% 
% open('ethyl_brainsketch.jpg')  %Use this same brainsketch for both animals
% 
% Npairs = zeros(1,7);
% for n=1:7
%     Npairs(n) = sum(links(:,3,n));
% end
% figure()
% plot(Npairs)
% 
% for n=1:1
%     figure
%     imshow(ethyl_brainsketch)
%     hold on
%     % for i = 1:size(xy,1)
%     %
%     %     text(xy(i,1),xy(i,2),num2str(i))
%     %
%     %     plot(xy(i,1),xy(i,2),'o')
%     %
%     % end
%     
%     idx = find(links(:,3,n)==1);
%     ch1 = links(idx,1,n);
%     ch2 = links(idx,2,n);
%     
%     for i = 1:length(idx)
%         p1 = [xy(ch1(i),1), xy(ch2(i),1)];
%         p2 = [xy(ch1(i),2), xy(ch2(i),2)];
%         %plot(xy(ch1(i),1), xy(ch1(i),2), 'or')
%         %plot(xy(ch2(i),1), xy(ch2(i),2), '^g')
%         line(p1,p2)
%         plot(xy(ch1(i),1), xy(ch1(i),2), 'or')
%         plot(xy(ch2(i),1), xy(ch2(i),2), 'or')
%     end
% end

load('lucy_brainsketch_xy.mat')  %for lucy. For Ethyl it would be load('ethyl_bmf_grid.mat')

open('ethyl_brainsketch.jpg')  %Use this same brainsketch for both animals

figure ; hold on

imshow(ethyl_brainsketch)

for i = 1:size(xy,1) 

    text(xy(i,1),xy(i,2),num2str(i))

    %plot(xy(i,1),xy(i,2),'o')

end