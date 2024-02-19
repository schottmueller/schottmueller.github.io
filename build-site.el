(require 'package)
(setq package-user-dir (expand-file-name "./.packages"))
(setq package-archives '(("melpa" . "https://melpa.org/packages/")
                         ("elpa" . "https://elpa.gnu.org/packages/")))

;; Initialize the package system
(package-initialize)
(unless package-archive-contents
  (package-refresh-contents))

;; Install dependencies
;allows fontified source code blocks
(package-install 'htmlize)
;;org-mode citations
(package-install 'citeproc-org)

(require 'ox-publish)
 
  (setq my-blog-header-file "./header.html"
        my-blog-footer-file "./footer.html"
        org-html-validation-link nil)

  ;; Load partials on memory
  (defun my-blog-header (arg)
    (with-temp-buffer
      (insert-file-contents my-blog-header-file)
      (buffer-string)))

  (defun my-blog-footer (arg)
    (with-temp-buffer
      (insert-file-contents my-blog-footer-file)
      (buffer-string)))

  (setq org-publish-project-alist
        '(;; Publish the posts
          ("website-notes"
           :base-directory "./content"
           :base-extension "org"
           :publishing-directory "./public"
           :recursive t
           :publishing-function org-html-publish-to-html
           :headline-levels 4
           :section-numbers nil
           :html-head nil
           :html-head-include-default-style nil
           :html-head-include-scripts nil
	   :html-head-extra "<link rel=\"stylesheet\" href=\"./css/stylesheet.css\">"
           :html-preamble my-blog-header
           :html-postamble my-blog-footer
           )

          ;; For static files that should remain untouched
          ("website-static"
           :base-directory "./static"
           :base-extension "css\\|js\\|png\\|jpg\\|gif\\|pdf\\|jl\\|odp\\|py\\|nb\\|asc\\|ttf\\|tex\\|bib\\|zip\\|csl"
           :publishing-directory "./public"
           :recursive t
           :publishing-function org-publish-attachment
           )

          ;; Combine the two previous components in a single one
          ("website" :components ("website-notes" "website-static"))))

;; Generate the site output
(org-publish-all t)

(message "Build complete!")
