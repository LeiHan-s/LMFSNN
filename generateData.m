function data = generateData5(nii, nb)
    r = @(t) (1 + cos(4 * t / 2).^2);
    
    % the original boundary collocation nodes where the BCs are implemented
    t = 2 * pi * (1:nb)'/nb;
    [bx, by] = pol2cart(t, r(t));

    bx1 = bx(1:2:end);bx2 = bx(2:2:end); by1 = by(1:2:end);by2 = by(2:2:end);

    index2 = 1:3:length(bx); index1 = setdiff(1:length(bx), index2);
    bx1 = bx(index1);by1 = by(index1);bx2 = bx(index2);by2 = by(index2);
    
    a = min(bx);
    b = max(bx);
    c1 = min(by);
    d = max(by);
    
    p = haltonset(2, 'Skip', 1500);
    q = net(p, 50000);
    q(:, 1) = q(:, 1) * (b - a) + a;
    q(:, 2) = q(:, 2) * (d - c1) + c1;
    pts = q;
    
    in = inpolygon(pts(:, 1), pts(:, 2), bx, by);
    pts = pts(in, :);
    pts = pts(1:nii, :);
    xii = pts(1:nii, 1);
    yii = pts(1:nii, 2);
    
    % the xi should include the second set of boundary points (bx2,by2) rather than (bx,by)
    xi = [bx1; xii];
    yi = [by1; yii];
    x = [xi; bx2];
    y = [yi; by2];
    data = [x, y];
end