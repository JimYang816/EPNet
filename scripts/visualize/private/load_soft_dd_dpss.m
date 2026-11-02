function [curve, nmseDb] = load_soft_dd_dpss(report, reportPath)
% Read the soft-decision result even when an older curve shares its internal ID.
    key = matlab.lang.makeValidName('DD-DPSS');
    assert(isfield(report.curves, key) && strcmp(report.curves.(key).identifier, 'dd_ce_lmmse') ...
        && isfield(report.nmse, 'dd_mean_nmse_db'), 'ICCPlot:MissingDDDPSS', ...
        'Expected DD-DPSS BER and corrected CSI NMSE in %s.', reportPath);
    curve = report.curves.(key);
    nmseDb = report.nmse.dd_mean_nmse_db(:).';

    auditPath = fullfile(fileparts(fileparts(reportPath)), 'verification.json');
    if isfile(auditPath)
        audit = jsondecode(fileread(auditPath));
        checked = audit.channels.(report.channel);
        assert(isfield(checked, 'feedback') && startsWith(checked.feedback, 'QPSK posterior mean') ...
            && strcmp(curve.provenance.cache_key, checked.dd_cache_key) ...
            && isequal(curve.ber(:), checked.dd_ber(:)) && isequal(nmseDb(:), checked.dd_mean_nmse_db(:)), ...
            'ICCPlot:DDDPSSVersion', 'DD-DPSS differs from the verified soft-decision result in %s.', reportPath);
    else
        % Standalone evaluator reports carry the numerical source fingerprint.
        detector = matlab.lang.makeValidName('src.algorithms.dd_ce_lmmse.detector');
        softSource = '2061c8fd667c4c5d412276cdb2fe9b7cc3704f04ca0d1d3dedcd0a1eb0ddd0b9'; % 2f94c6c
        assert(strcmp(curve.provenance.identity.code.(detector), softSource), 'ICCPlot:DDDPSSVersion', ...
            'Expected the soft-decision implementation in %s.', reportPath);
    end
end
