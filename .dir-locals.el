;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((python-mode
  (eval pyvenv-activate
	(shell-command-to-string "poetry env info --path"))))
