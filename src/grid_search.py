from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import f1_score


def grid_search_reporter(models_w_feature_dict, train_x, train_y):
    searches = grid_search_all(models_w_feature_dict, train_x, train_y)

    for gs in searches:
        print "====={}=====".format(gs.best_estimator_.__class__.__name__)
        print " recall_score: {}".format(gs.best_score_)
        print "  params: {}".format(gs.best_params_)

    return searches


def grid_search_all(models_w_feature_dict, train_x, train_y):
    grid_searches = []
    for model, feature_dict in models_w_feature_dict:
        gs = grid_search(model, feature_dict, train_x, train_y)
        grid_searches.append(gs)

    return grid_searches


def grid_search(model, feature_dict, train_x, train_y):
    gscv = GridSearchCV(
        model,
        feature_dict,
        n_jobs=-1,
        verbose=True,
        scoring='recall'
    )

    gscv.fit(train_x, train_y)
    return gscv
