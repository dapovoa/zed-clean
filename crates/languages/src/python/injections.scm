((comment) @injection.content
 (#set! injection.language "comment")
)

; SQL -----------------------------------------------------------------------------
(
    [
        ; function calls
        (call
            [
                (attribute attribute: (identifier) @_function_name)
                (identifier) @_function_name
            ]
            arguments: (argument_list
                (comment) @_comment
                (string
                    (string_content) @injection.content
                )
        ))

        ; string variables
        ((comment) @_comment
            .
            (expression_statement
                (assignment
                    right: (string
                        (string_content) @injection.content
                    )
                )
        ))
    ]
    (#match? @_comment "^(#|#\\s+)(?i:sql)\\s*$")
    (#set! injection.language "sql")
)
